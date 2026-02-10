"""
Polymarket Near-Expiry Arbitrage Scanner
=========================================
Scans for markets expiring within a configurable time window, identifies
high-probability outcomes (92-99c range), analyzes order book depth, and
calculates expected profit from lifting the book.

Strategy premise:
  - A market trading at 95c with 2 hours to expiry is almost certainly
    going to resolve YES. Buy shares at 95c, collect $1 at resolution.
  - Profit = (1.00 - avg_fill_price) * shares - fees
  - Risk = the "almost certain" outcome flips and you lose your cost basis.

Usage:
  pip install requests tabulate
  python scanner.py

Optional flags:
  python scanner.py --hours 6 --min-price 0.90 --max-price 0.99 --refresh 60
"""

import argparse
import io
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

# Fix Windows console encoding for market names with special characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

try:
    import requests
except ImportError:
    print("Missing dependency. Run: pip install requests")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    print("Missing dependency. Run: pip install tabulate")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_API = "https://clob.polymarket.com"

# Polymarket charges ~2% on winnings (not on the full notional).
# Specifically: fee is on profit, i.e. (payout - cost). As of 2025 the
# fee rate is approximately 2% but can vary. We use 2% as a conservative
# estimate. Adjust if Polymarket updates their fee schedule.
FEE_RATE = 0.02

# Minimum order size on Polymarket CLOB (in shares / USDC notional)
MIN_ORDER_SIZE = 5.0

# Rate limit: be polite to the API
REQUEST_DELAY = 0.15  # seconds between CLOB requests

# ---------------------------------------------------------------------------
# Sports Filter
# ---------------------------------------------------------------------------
# Keywords that indicate a sports-related market (case-insensitive match
# against the market question / slug). Add or remove terms as needed.
SPORTS_KEYWORDS = [
    # Major leagues & events
    "super bowl", "nfl", "nba", "mlb", "nhl", "mls", "wnba",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "champions league", "europa league", "world cup", "fifa",
    "olympic", "olympics", "grand slam", "grand prix", "formula 1",
    "uefa", "playoff", "playoffs",
    # Sports-specific terms
    "touchdown", "field goal", "quarterback", "rushing yards",
    "receiving yards", "passing yards", "halftime show",
    "gatorade shower", "first touchdown",
    "home run", "strikeout", "pitcher",
    "three-pointer", "free throw",
    "hat trick", "penalty kick", "corner kick", "red card",
    "yellow card", "clean sheet",
    # Sports types
    "soccer", "football", "basketball", "baseball", "hockey",
    "tennis", "golf", "boxing", "ufc", "mma", "wrestling",
    "cricket", "rugby", "volleyball", "handball",
    # Esports
    "esports", "counter-strike", "csgo", "cs2", "dota 2", "valorant",
    "league of legends", "overwatch", "call of duty",
    # Sports betting patterns
    "moneyline", "parlay",
    "map winner", "map 1 winner", "map 2 winner", "map 3 winner",
    "(bo3)", "(bo5)", "(bo1)",
    # Player stat lines
    "rushing yard", "receiving yard", "passing yard",
    "tackles", "sacks", "interceptions",
    # Sports-specific context
    "national anthem",
]

def is_sports_market(question: str, slug: str = "") -> bool:
    """Return True if the market question/slug matches sports keywords."""
    text = (question + " " + slug).lower()
    return any(kw in text for kw in SPORTS_KEYWORDS)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class OrderLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    bids: list[OrderLevel] = field(default_factory=list)
    asks: list[OrderLevel] = field(default_factory=list)
    tick_size: float = 0.001
    min_order_size: float = 5.0


@dataclass
class MarketInfo:
    """Represents a single Polymarket binary outcome market."""
    market_id: str
    question: str
    slug: str
    end_date: datetime
    outcomes: list[str]
    outcome_prices: list[float]
    clob_token_ids: list[str]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    volume: float
    liquidity: float
    last_trade_price: Optional[float]
    neg_risk: bool = False

    @property
    def time_to_expiry(self) -> timedelta:
        return self.end_date - datetime.now(timezone.utc)

    @property
    def hours_to_expiry(self) -> float:
        return self.time_to_expiry.total_seconds() / 3600

    @property
    def yes_price(self) -> Optional[float]:
        if self.outcome_prices and len(self.outcome_prices) >= 1:
            return self.outcome_prices[0]
        return None

    @property
    def no_price(self) -> Optional[float]:
        if self.outcome_prices and len(self.outcome_prices) >= 2:
            return self.outcome_prices[1]
        return None


@dataclass
class LiftAnalysis:
    """Result of analyzing what happens if you buy up available shares."""
    side: str                   # "YES" or "NO"
    token_id: str
    levels_consumed: int        # how many price levels you'd eat through
    total_shares: float         # total shares you could buy
    total_cost: float           # total USDC spent
    avg_fill_price: float       # volume-weighted average price
    worst_fill_price: float     # highest price you'd pay
    gross_profit: float         # (1.0 - avg_fill) * shares
    fee: float                  # fee on profit
    net_profit: float           # gross - fee
    roi_pct: float              # net_profit / total_cost * 100
    fills: list[tuple[float, float]]  # (price, size) for each level


# ---------------------------------------------------------------------------
# API Layer
# ---------------------------------------------------------------------------

def fetch_markets_page(
    end_date_min: str,
    end_date_max: str,
    offset: int = 0,
    limit: int = 100,
) -> list[dict]:
    """Fetch one page of active markets from the Gamma API."""
    params = {
        "limit": limit,
        "offset": offset,
        "active": "true",
        "closed": "false",
        "end_date_min": end_date_min,
        "end_date_max": end_date_max,
        "order": "endDate",
        "ascending": "true",
    }
    resp = requests.get(GAMMA_API, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all_markets(end_date_min: str, end_date_max: str) -> list[dict]:
    """Paginate through the Gamma API to get all matching markets."""
    all_markets = []
    offset = 0
    page_size = 100

    while True:
        page = fetch_markets_page(end_date_min, end_date_max, offset, page_size)
        if not page:
            break
        all_markets.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(REQUEST_DELAY)

    return all_markets


def fetch_order_book(token_id: str) -> OrderBook:
    """Fetch the full order book for a single token from the CLOB API."""
    resp = requests.get(
        f"{CLOB_API}/book",
        params={"token_id": token_id},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    bids = [
        OrderLevel(price=float(b["price"]), size=float(b["size"]))
        for b in data.get("bids", [])
    ]
    asks = [
        OrderLevel(price=float(a["price"]), size=float(a["size"]))
        for a in data.get("asks", [])
    ]

    # Sort bids descending (best bid first), asks ascending (best ask first)
    bids.sort(key=lambda x: x.price, reverse=True)
    asks.sort(key=lambda x: x.price)

    return OrderBook(
        bids=bids,
        asks=asks,
        tick_size=float(data.get("tick_size", "0.001")),
        min_order_size=float(data.get("min_order_size", "5")),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_market(raw: dict) -> Optional[MarketInfo]:
    """Convert raw Gamma API JSON into a MarketInfo object."""
    try:
        end_date_str = raw.get("endDate", "")
        if not end_date_str:
            return None

        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))

        clob_ids_raw = raw.get("clobTokenIds", "")
        if isinstance(clob_ids_raw, str):
            # Sometimes returned as a JSON string like '["id1","id2"]'
            import json
            try:
                clob_ids = json.loads(clob_ids_raw)
            except (json.JSONDecodeError, TypeError):
                clob_ids = []
        else:
            clob_ids = clob_ids_raw or []

        outcome_prices_raw = raw.get("outcomePrices", "")
        if isinstance(outcome_prices_raw, str):
            import json
            try:
                outcome_prices = [float(p) for p in json.loads(outcome_prices_raw)]
            except (json.JSONDecodeError, TypeError, ValueError):
                outcome_prices = []
        elif isinstance(outcome_prices_raw, list):
            outcome_prices = [float(p) for p in outcome_prices_raw]
        else:
            outcome_prices = []

        outcomes_raw = raw.get("outcomes", "")
        if isinstance(outcomes_raw, str):
            import json
            try:
                outcomes = json.loads(outcomes_raw)
            except (json.JSONDecodeError, TypeError):
                outcomes = ["Yes", "No"]
        else:
            outcomes = outcomes_raw or ["Yes", "No"]

        def safe_float(val):
            if val is None or val == "":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        return MarketInfo(
            market_id=str(raw.get("id", "")),
            question=raw.get("question", ""),
            slug=raw.get("slug", ""),
            end_date=end_date,
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_ids,
            best_bid=safe_float(raw.get("bestBid")),
            best_ask=safe_float(raw.get("bestAsk")),
            spread=safe_float(raw.get("spread")),
            volume=float(raw.get("volumeNum", 0) or raw.get("volume", 0) or 0),
            liquidity=float(raw.get("liquidityNum", 0) or raw.get("liquidity", 0) or 0),
            last_trade_price=safe_float(raw.get("lastTradePrice")),
            neg_risk=bool(raw.get("negRisk", False)),
        )
    except Exception as e:
        print(f"  [warn] Failed to parse market {raw.get('id', '?')}: {e}")
        return None


# ---------------------------------------------------------------------------
# Analysis Engine
# ---------------------------------------------------------------------------

def analyze_lift(
    order_book: OrderBook,
    side: str,
    token_id: str,
    max_price: float = 0.99,
    max_spend: float = 10_000.0,
) -> Optional[LiftAnalysis]:
    """
    Simulate lifting the order book up to max_price.

    For a BUY (going long YES or NO), we walk up the ask side.
    We buy every ask level at or below max_price until we either
    exhaust the book or hit our max_spend budget.

    Args:
        order_book: The full order book
        side: "YES" or "NO" -- which outcome token we're buying
        token_id: The CLOB token ID for this side
        max_price: Don't buy above this price (e.g., 0.99)
        max_spend: Maximum USDC to deploy
    """
    # We want to buy asks (lift the ask side)
    eligible_asks = [a for a in order_book.asks if a.price <= max_price]

    if not eligible_asks:
        return None

    fills = []
    total_shares = 0.0
    total_cost = 0.0
    worst_price = 0.0

    for level in eligible_asks:
        level_cost = level.price * level.size
        remaining_budget = max_spend - total_cost

        if remaining_budget <= 0:
            break

        if level_cost <= remaining_budget:
            # Take the full level
            fills.append((level.price, level.size))
            total_shares += level.size
            total_cost += level_cost
            worst_price = level.price
        else:
            # Partial fill on this level
            affordable_shares = remaining_budget / level.price
            fills.append((level.price, affordable_shares))
            total_shares += affordable_shares
            total_cost += affordable_shares * level.price
            worst_price = level.price
            break

    if total_shares == 0 or total_cost == 0:
        return None

    avg_fill = total_cost / total_shares

    # If the market resolves to this outcome, each share pays $1
    # Profit per share = 1.0 - avg_fill_price
    # Polymarket fee is on the PROFIT portion only
    gross_profit = (1.0 - avg_fill) * total_shares
    fee = gross_profit * FEE_RATE
    net_profit = gross_profit - fee
    roi_pct = (net_profit / total_cost) * 100 if total_cost > 0 else 0

    return LiftAnalysis(
        side=side,
        token_id=token_id,
        levels_consumed=len(fills),
        total_shares=total_shares,
        total_cost=total_cost,
        avg_fill_price=avg_fill,
        worst_fill_price=worst_price,
        gross_profit=gross_profit,
        fee=fee,
        net_profit=net_profit,
        roi_pct=roi_pct,
        fills=fills,
    )


def is_in_target_range(price: Optional[float], min_price: float, max_price: float) -> bool:
    """Check if a price falls within our target high-probability range."""
    if price is None:
        return False
    return min_price <= price <= max_price


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "EXPIRED"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


def display_results(
    markets: list[MarketInfo],
    analyses: dict[str, list[LiftAnalysis]],
    min_price: float,
    max_price: float,
):
    """Print the scan results in a readable format."""
    now = datetime.now(timezone.utc)

    print("\n" + "=" * 100)
    print(f"  POLYMARKET NEAR-EXPIRY SCANNER")
    print(f"  Scan time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Target price range: {min_price:.0%} - {max_price:.0%}")
    print(f"  Fee assumption: {FEE_RATE:.1%} on profits")
    print("=" * 100)

    if not markets:
        print("\n  No markets found matching criteria.\n")
        return

    # ---- Summary Table ----
    summary_rows = []
    for m in markets:
        yes_flag = "*" if is_in_target_range(m.yes_price, min_price, max_price) else ""
        no_flag = "*" if is_in_target_range(m.no_price, min_price, max_price) else ""

        summary_rows.append([
            m.market_id,
            m.question[:60] + ("..." if len(m.question) > 60 else ""),
            format_timedelta(m.time_to_expiry),
            f"{m.yes_price:.1%}" if m.yes_price else "N/A",
            f"{m.no_price:.1%}" if m.no_price else "N/A",
            f"{m.best_bid:.3f}" if m.best_bid else "N/A",
            f"{m.best_ask:.3f}" if m.best_ask else "N/A",
            f"${m.volume:,.0f}",
            f"${m.liquidity:,.0f}",
            yes_flag + no_flag,
        ])

    print("\n-- All Near-Expiry Markets --\n")
    print(tabulate(
        summary_rows,
        headers=["ID", "Question", "Expiry", "YES", "NO", "Bid", "Ask",
                 "Volume", "Liq", "Flag"],
        tablefmt="simple",
    ))
    print(f"\n  * = outcome price in target range ({min_price:.0%}-{max_price:.0%})\n")

    # ---- Detailed Lift Analysis ----
    flagged = [m for m in markets
               if is_in_target_range(m.yes_price, min_price, max_price)
               or is_in_target_range(m.no_price, min_price, max_price)]

    if not flagged:
        print("  No markets in target price range. Try widening the range or window.\n")
        return

    print("\n" + "=" * 100)
    print("  ORDER BOOK LIFT ANALYSIS (flagged markets)")
    print("=" * 100)

    for m in flagged:
        market_analyses = analyses.get(m.market_id, [])
        if not market_analyses:
            continue

        print(f"\n  Market: {m.question}")
        print(f"  ID: {m.market_id} | Expiry: {format_timedelta(m.time_to_expiry)}")
        print(f"  Slug: https://polymarket.com/event/{m.slug}")
        print(f"  YES={m.yes_price:.1%}  NO={m.no_price:.1%}")
        print(f"  Best Bid={m.best_bid}  Best Ask={m.best_ask}  Spread={m.spread}")
        print()

        for la in market_analyses:
            if la.total_shares == 0:
                continue

            print(f"  --- BUY {la.side} tokens ---")
            print(f"  Avg fill:       {la.avg_fill_price:.4f} ({la.avg_fill_price:.2%})")
            print(f"  Worst fill:     {la.worst_fill_price:.4f}")
            print(f"  Levels eaten:   {la.levels_consumed}")
            print(f"  Total shares:   {la.total_shares:,.2f}")
            print(f"  Total cost:     ${la.total_cost:,.2f}")
            print(f"  Gross profit:   ${la.gross_profit:,.2f}")
            print(f"  Fee ({FEE_RATE:.0%}):       ${la.fee:,.2f}")
            print(f"  Net profit:     ${la.net_profit:,.2f}")
            print(f"  ROI:            {la.roi_pct:.2f}%")

            # Show top-of-book detail (first 5 levels)
            print(f"\n  Top ask levels consumed:")
            top_n = la.fills[:8]
            fill_rows = [
                [f"{price:.4f}", f"{size:,.2f}", f"${price * size:,.2f}"]
                for price, size in top_n
            ]
            print(tabulate(
                fill_rows,
                headers=["Price", "Size (shares)", "Cost (USDC)"],
                tablefmt="simple",
                stralign="right",
            ))
            if len(la.fills) > 8:
                print(f"  ... and {len(la.fills) - 8} more levels")
            print()

    # ---- Risk Warning ----
    print("-" * 100)
    print("  RISK WARNING:")
    print("  - These profits assume the high-probability outcome resolves as expected.")
    print("  - If it does NOT, you lose your entire cost basis.")
    print("  - Near-expiry markets can have resolution delays or disputes.")
    print("  - Order book depth shown is a snapshot; it can change by the time you trade.")
    print("  - Always verify the resolution source and consider the tail risk.")
    print("-" * 100)
    print()


# ---------------------------------------------------------------------------
# Main Scanner Loop
# ---------------------------------------------------------------------------

def run_scan(
    hours_window: float = 12.0,
    min_price: float = 0.92,
    max_price: float = 0.99,
    max_spend: float = 10_000.0,
    verbose: bool = False,
):
    """Execute one full scan cycle."""
    now = datetime.now(timezone.utc)
    end_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_max = (now + timedelta(hours=hours_window)).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n[{now.strftime('%H:%M:%S UTC')}] Scanning for markets expiring between now and {hours_window}h from now...")

    # Step 1: Fetch all near-expiry markets from Gamma API
    raw_markets = fetch_all_markets(end_min, end_max)
    print(f"  Found {len(raw_markets)} raw markets from Gamma API")

    # Step 2: Parse into structured objects
    markets = []
    for raw in raw_markets:
        m = parse_market(raw)
        if m is not None and m.clob_token_ids:
            markets.append(m)

    print(f"  Parsed {len(markets)} valid markets with CLOB token IDs")

    # Filter out sports-related markets
    before = len(markets)
    markets = [m for m in markets if not is_sports_market(m.question, m.slug)]
    if before != len(markets):
        print(f"  Filtered out {before - len(markets)} sports markets -> {len(markets)} remaining")

    if not markets:
        print("  No near-expiry markets found. Try a wider time window (--hours).")
        return

    # Step 3: Identify flagged markets (in target price range)
    flagged = [m for m in markets
               if is_in_target_range(m.yes_price, min_price, max_price)
               or is_in_target_range(m.no_price, min_price, max_price)]

    print(f"  {len(flagged)} markets in target price range ({min_price:.0%}-{max_price:.0%})")

    # Step 4: Fetch order books for flagged markets and run lift analysis
    analyses: dict[str, list[LiftAnalysis]] = {}

    for m in flagged:
        market_lifts = []

        for i, (outcome, price) in enumerate(zip(m.outcomes, m.outcome_prices)):
            if not is_in_target_range(price, min_price, max_price):
                continue
            if i >= len(m.clob_token_ids):
                continue

            token_id = m.clob_token_ids[i]

            try:
                if verbose:
                    print(f"  Fetching order book for {outcome} side of market {m.market_id}...")
                ob = fetch_order_book(token_id)
                time.sleep(REQUEST_DELAY)

                la = analyze_lift(ob, outcome, token_id, max_price=max_price, max_spend=max_spend)
                if la is not None:
                    market_lifts.append(la)
            except requests.RequestException as e:
                print(f"  [warn] Failed to fetch order book for {m.market_id}/{outcome}: {e}")
                continue

        if market_lifts:
            analyses[m.market_id] = market_lifts

    # Step 5: Display everything
    display_results(markets, analyses, min_price, max_price)


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Near-Expiry Arbitrage Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scanner.py                          # Default: 12h window, 92-99c range
  python scanner.py --hours 6                # Only markets expiring in next 6 hours
  python scanner.py --min-price 0.95         # Tighter range: 95-99c
  python scanner.py --max-spend 5000         # Limit simulated spend to $5,000
  python scanner.py --refresh 60             # Re-scan every 60 seconds
  python scanner.py --hours 24 --verbose     # Wider window with debug output
        """,
    )
    parser.add_argument(
        "--hours", type=float, default=2.0,
        help="Time window in hours to look for expiring markets (default: 2)",
    )
    parser.add_argument(
        "--min-price", type=float, default=0.92,
        help="Minimum outcome price to flag (default: 0.92)",
    )
    parser.add_argument(
        "--max-price", type=float, default=0.99,
        help="Maximum outcome price to flag (default: 0.99)",
    )
    parser.add_argument(
        "--max-spend", type=float, default=10_000.0,
        help="Max USDC to simulate spending per market side (default: 10000)",
    )
    parser.add_argument(
        "--refresh", type=int, default=180,
        help="Re-scan interval in seconds. 0 = single scan (default: 180)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print extra debug info during scan",
    )

    args = parser.parse_args()

    # Validate inputs
    if not (0.0 < args.min_price < 1.0):
        print("Error: --min-price must be between 0 and 1")
        sys.exit(1)
    if not (0.0 < args.max_price <= 1.0):
        print("Error: --max-price must be between 0 and 1")
        sys.exit(1)
    if args.min_price >= args.max_price:
        print("Error: --min-price must be less than --max-price")
        sys.exit(1)

    print("Polymarket Near-Expiry Arbitrage Scanner")
    print(f"  Window:      {args.hours}h")
    print(f"  Price range: {args.min_price:.0%} - {args.max_price:.0%}")
    print(f"  Max spend:   ${args.max_spend:,.0f}")
    print(f"  Refresh:     {'single scan' if args.refresh == 0 else f'every {args.refresh}s'}")

    if args.refresh > 0:
        print("\n  Press Ctrl+C to stop.\n")
        try:
            while True:
                run_scan(
                    hours_window=args.hours,
                    min_price=args.min_price,
                    max_price=args.max_price,
                    max_spend=args.max_spend,
                    verbose=args.verbose,
                )
                print(f"\n  Next scan in {args.refresh} seconds...")
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\n  Scanner stopped.")
    else:
        run_scan(
            hours_window=args.hours,
            min_price=args.min_price,
            max_price=args.max_price,
            max_spend=args.max_spend,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
