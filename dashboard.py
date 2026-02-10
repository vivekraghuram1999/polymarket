"""
Polymarket Near-Expiry Dashboard (Dash + Plotly)
=================================================
Two tabs:
  1. Live Scanner - real-time near-expiry market scanner with order book depth
  2. Paper Trading - auto-places best trades from scanner, monitors positions

Usage:
  pip install dash plotly requests pandas
  python dashboard.py
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    import requests
except ImportError:
    print("Run: pip install requests")
    sys.exit(1)

try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
except ImportError:
    print("Run: pip install dash plotly")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Run: pip install pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_API = "https://clob.polymarket.com"
FEE_RATE = 0.02
REQUEST_DELAY = 0.12
MAX_PORTFOLIO = 1000  # total max invested across all open positions
MAX_PER_TRADE = 100   # max allocation per individual trade
STOP_LOSS_PCT = 0.20  # default stop-loss: sell if position drops this % from entry

COLORS = {
    "bg": "#0d1117", "card": "#161b22", "border": "#30363d",
    "text": "#e6edf3", "muted": "#8b949e", "green": "#3fb950",
    "red": "#f85149", "yellow": "#d29922", "blue": "#58a6ff", "purple": "#bc8cff",
}

# ---------------------------------------------------------------------------
# Sports Filter
# ---------------------------------------------------------------------------
SPORTS_KEYWORDS = [
    "super bowl", "nfl", "nba", "mlb", "nhl", "mls", "wnba",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "champions league", "europa league", "world cup", "fifa",
    "olympic", "olympics", "grand slam", "grand prix", "formula 1",
    "uefa", "playoff", "playoffs",
    "touchdown", "field goal", "quarterback", "rushing yards",
    "receiving yards", "passing yards", "halftime show",
    "gatorade shower", "first touchdown",
    "home run", "strikeout", "pitcher",
    "three-pointer", "free throw",
    "hat trick", "penalty kick", "corner kick", "red card",
    "yellow card", "clean sheet",
    "soccer", "football", "basketball", "baseball", "hockey",
    "tennis", "golf", "boxing", "ufc", "mma", "wrestling",
    "cricket", "rugby", "volleyball", "handball",
    "esports", "counter-strike", "csgo", "cs2", "dota 2", "valorant",
    "league of legends", "overwatch", "call of duty",
    "moneyline", "parlay",
    "map winner", "map 1 winner", "map 2 winner", "map 3 winner",
    "(bo3)", "(bo5)", "(bo1)",
    "rushing yard", "receiving yard", "passing yard",
    "tackles", "sacks", "interceptions",
    "national anthem",
]

def is_sports_market(question, slug=""):
    text = (question + " " + slug).lower()
    return any(kw in text for kw in SPORTS_KEYWORDS)
CELL_STYLE = {
    "backgroundColor": COLORS["card"], "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}", "padding": "6px 8px",
    "fontSize": "13px", "whiteSpace": "normal",
}
HEADER_STYLE = {
    "backgroundColor": COLORS["bg"], "color": COLORS["text"],
    "fontWeight": "bold", "border": f"1px solid {COLORS['border']}",
    "fontSize": "12px", "padding": "6px 8px",
}

# ---------------------------------------------------------------------------
# Paper Trade Persistence
# ---------------------------------------------------------------------------
# Vercel has a read-only filesystem; use /tmp there
if os.environ.get("VERCEL"):
    PAPER_TRADES_FILE = "/tmp/paper_trades.json"
else:
    PAPER_TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades.json")


def load_paper_trades():
    if not os.path.exists(PAPER_TRADES_FILE):
        return []
    try:
        with open(PAPER_TRADES_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_paper_trades(trades):
    with open(PAPER_TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def add_paper_trade(market_id, question, clob_id, side, entry_price, shares, cost, expiry_time, book_snapshot=None):
    trades = load_paper_trades()
    trade = {
        "trade_id": str(uuid.uuid4())[:8],
        "market_id": str(market_id),
        "question": question,
        "clob_id": clob_id,
        "side": side,
        "entry_price": entry_price,
        "shares": shares,
        "cost": cost,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "expiry_time": expiry_time,
        "status": "open",
        "resolution": None,
        "payout": None,
        "net_pnl": None,
        "resolved_at": None,
        "book_snapshot": book_snapshot or {},
    }
    trades.append(trade)
    save_paper_trades(trades)
    return trade


def auto_place_trades(flagged_rows):
    """Auto-place paper trades for GOOD and OK verdict scanner results.
    Skips markets that already have an open trade on the same side.
    Respects MAX_PORTFOLIO cap on total invested across all open positions."""
    tradeable = [r for r in flagged_rows if r.get("verdict") in ("GOOD", "OK")
                 and r.get("_roi", 0) > 0 and r.get("_depth", 0) > 0]
    if not tradeable:
        return 0
    existing = load_paper_trades()
    open_keys = {(t["market_id"], t["side"]) for t in existing if t["status"] == "open"}
    total_invested = sum(t["cost"] for t in existing if t["status"] == "open")
    placed = 0
    for row in tradeable:
        remaining = MAX_PORTFOLIO - total_invested
        if remaining < 1.0:
            break
        market_id = row.get("id", "")
        side = row.get("side", "YES").upper()
        if (market_id, side) in open_keys:
            continue
        try:
            avg_fill_str = row.get("avg_fill", "N/A").replace("%", "")
            avg_fill = float(avg_fill_str) / 100.0 if avg_fill_str != "N/A" else None
        except (ValueError, TypeError):
            avg_fill = None
        try:
            shares_str = row.get("max_buy_shares", "0").replace(",", "")
            shares = float(shares_str)
        except (ValueError, TypeError):
            shares = 0
        if not avg_fill or shares <= 0:
            continue
        cost = round(avg_fill * shares, 4)
        # Cap each trade to MAX_PER_TRADE and remaining portfolio budget
        trade_cap = min(MAX_PER_TRADE, remaining)
        if cost > trade_cap:
            shares = round(trade_cap / avg_fill, 2)
            cost = round(avg_fill * shares, 4)
        if shares <= 0 or cost < 1.0:
            continue
        clob_id = row.get("_clob_id", "")
        question = row.get("question", "")
        expiry = row.get("_end_date", "")  # store actual ISO end_date, not display string
        book_snapshot = {
            "best_bid": row.get("best_bid", "N/A"),
            "best_ask": row.get("best_ask", "N/A"),
            "spread": row.get("spread", "N/A"),
            "book_depth_shares": row.get("book_depth_shares", "0"),
            "book_depth_usd": row.get("book_depth_usd", "$0"),
            "worst_fill": row.get("worst_fill", "N/A"),
            "levels_eaten": row.get("levels_eaten", "0"),
            "bid_depth_usd": row.get("bid_depth_usd", "$0"),
            "verdict": row.get("verdict", ""),
            "roi_at_entry": row.get("roi", "0%"),
            "net_profit_est": row.get("net_profit", "$0"),
        }
        add_paper_trade(market_id, question, clob_id, side, round(avg_fill, 4), round(shares, 2), cost, expiry, book_snapshot)
        open_keys.add((market_id, side))
        total_invested += cost
        placed += 1
    return placed


def check_and_resolve_trades(stop_loss_pct=None):
    if stop_loss_pct is None:
        stop_loss_pct = STOP_LOSS_PCT
    trades = load_paper_trades()
    open_trades = [t for t in trades if t["status"] == "open"]
    if not open_trades:
        return trades
    changed = set()
    for t in open_trades:
        try:
            resp = requests.get(GAMMA_API, params={"id": t["market_id"]}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue
            market = data[0] if isinstance(data, list) else data

            # Check market resolution first
            if market.get("resolved", False) and market.get("resolution", ""):
                resolution = market["resolution"]
                won = (t["side"] == "YES" and resolution == "Yes") or \
                      (t["side"] == "NO" and resolution == "No")
                payout = t["shares"] * 1.0 if won else 0.0
                gross_pnl = payout - t["cost"]
                fee = max(gross_pnl, 0) * FEE_RATE
                net_pnl = gross_pnl - fee
                t["status"] = "resolved_win" if won else "resolved_loss"
                t["resolution"] = resolution
                t["payout"] = round(payout, 4)
                t["net_pnl"] = round(net_pnl, 4)
                t["resolved_at"] = datetime.now(timezone.utc).isoformat()
                changed.add(t["trade_id"])
                time.sleep(REQUEST_DELAY)
                continue

            # Stop-loss check on unresolved positions
            if stop_loss_pct > 0:
                outcome_prices = parse_json_field(market.get("outcomePrices", ""))
                outcomes = parse_json_field(market.get("outcomes", "")) or ["Yes", "No"]
                price_map = {}
                for oc, pr in zip(outcomes, outcome_prices):
                    try:
                        price_map[oc.upper()] = float(pr)
                    except (ValueError, TypeError):
                        pass
                current_price = price_map.get(t["side"])
                if current_price is not None and t["entry_price"] > 0:
                    pct_change = (current_price - t["entry_price"]) / t["entry_price"]
                    if pct_change <= -stop_loss_pct:
                        payout = current_price * t["shares"]
                        net_pnl = payout - t["cost"]
                        t["status"] = "resolved_loss"
                        t["resolution"] = f"Stop Loss ({stop_loss_pct:.0%})"
                        t["payout"] = round(payout, 4)
                        t["net_pnl"] = round(net_pnl, 4)
                        t["resolved_at"] = datetime.now(timezone.utc).isoformat()
                        changed.add(t["trade_id"])

            time.sleep(REQUEST_DELAY)
        except Exception:
            continue
    if changed:
        save_paper_trades(trades)
    return trades


def get_current_prices(open_trades):
    """Fetch current outcome prices for open positions via Gamma API."""
    prices = {}
    seen_markets = {}
    for t in open_trades:
        mid = t["market_id"]
        if mid in seen_markets:
            prices[t["trade_id"]] = seen_markets[mid].get(t["side"])
            continue
        try:
            resp = requests.get(GAMMA_API, params={"id": mid}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue
            market = data[0] if isinstance(data, list) else data
            outcome_prices = parse_json_field(market.get("outcomePrices", ""))
            outcomes = parse_json_field(market.get("outcomes", "")) or ["Yes", "No"]
            price_map = {}
            for oc, pr in zip(outcomes, outcome_prices):
                try:
                    price_map[oc.upper()] = float(pr)
                except (ValueError, TypeError):
                    pass
            seen_markets[mid] = price_map
            prices[t["trade_id"]] = price_map.get(t["side"])
            time.sleep(REQUEST_DELAY)
        except Exception:
            continue
    return prices


# ---------------------------------------------------------------------------
# Shared API helpers
# ---------------------------------------------------------------------------

def fetch_markets_page(end_min, end_max, offset=0, limit=100, active="true", closed="false"):
    resp = requests.get(GAMMA_API, params={
        "limit": limit, "offset": offset, "active": active, "closed": closed,
        "end_date_min": end_min, "end_date_max": end_max,
        "order": "endDate", "ascending": "true",
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all_markets(end_min, end_max, active="true", closed="false"):
    all_m = []
    offset = 0
    while True:
        page = fetch_markets_page(end_min, end_max, offset, active=active, closed=closed)
        if not page:
            break
        all_m.extend(page)
        if len(page) < 100:
            break
        offset += 100
        time.sleep(REQUEST_DELAY)
    return all_m


def fetch_order_book(token_id):
    resp = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def safe_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_json_field(raw_val):
    if isinstance(raw_val, list):
        return raw_val
    if isinstance(raw_val, str):
        try:
            return json.loads(raw_val)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def parse_market(raw):
    try:
        end_str = raw.get("endDate", "")
        if not end_str:
            return None
        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        clob_ids = parse_json_field(raw.get("clobTokenIds", ""))
        outcomes = parse_json_field(raw.get("outcomes", "")) or ["Yes", "No"]
        prices_raw = parse_json_field(raw.get("outcomePrices", ""))
        prices = [float(p) for p in prices_raw] if prices_raw else []
        return {
            "id": str(raw.get("id", "")),
            "question": raw.get("question", ""),
            "slug": raw.get("slug", ""),
            "end_date": end_date,
            "outcomes": outcomes,
            "prices": prices,
            "clob_ids": clob_ids,
            "best_bid": safe_float(raw.get("bestBid")),
            "best_ask": safe_float(raw.get("bestAsk")),
            "spread": safe_float(raw.get("spread")),
            "volume": float(raw.get("volumeNum", 0) or raw.get("volume", 0) or 0),
            "liquidity": float(raw.get("liquidityNum", 0) or raw.get("liquidity", 0) or 0),
            "resolved": raw.get("resolved", False),
            "resolution": raw.get("resolution", ""),
        }
    except Exception:
        return None


def get_lift_analysis(order_book_data, max_price=0.99, max_spend=10000):
    asks_raw = order_book_data.get("asks", [])
    bids_raw = order_book_data.get("bids", [])
    asks = sorted([{"price": float(a["price"]), "size": float(a["size"])} for a in asks_raw], key=lambda x: x["price"])
    bids = sorted([{"price": float(b["price"]), "size": float(b["size"])} for b in bids_raw], key=lambda x: x["price"], reverse=True)
    eligible = [a for a in asks if a["price"] <= max_price]
    fills, total_shares, total_cost = [], 0.0, 0.0
    for lvl in eligible:
        remaining = max_spend - total_cost
        if remaining <= 0:
            break
        cost = lvl["price"] * lvl["size"]
        if cost <= remaining:
            fills.append(lvl)
            total_shares += lvl["size"]
            total_cost += cost
        else:
            partial = remaining / lvl["price"]
            fills.append({"price": lvl["price"], "size": partial})
            total_shares += partial
            total_cost += partial * lvl["price"]
            break
    if total_shares == 0:
        return None
    avg_fill = total_cost / total_shares
    gross = (1.0 - avg_fill) * total_shares
    fee = gross * FEE_RATE
    net = gross - fee
    roi = (net / total_cost) * 100
    range_depth = sum(a["size"] for a in eligible)
    range_value = sum(a["price"] * a["size"] for a in eligible)
    return {
        "fills": fills, "total_shares": total_shares, "total_cost": total_cost,
        "avg_fill": avg_fill, "worst_fill": fills[-1]["price"] if fills else 0,
        "gross_profit": gross, "fee": fee, "net_profit": net, "roi_pct": roi,
        "levels": len(fills), "total_ask_levels": len(asks),
        "total_ask_depth": sum(a["size"] for a in asks),
        "total_ask_value": sum(a["price"] * a["size"] for a in asks),
        "total_bid_depth": sum(b["size"] for b in bids),
        "total_bid_value": sum(b["price"] * b["size"] for b in bids),
        "range_depth": range_depth, "range_value": range_value,
        "best_ask": asks[0]["price"] if asks else None,
        "best_bid": bids[0]["price"] if bids else None,
        "spread": (asks[0]["price"] - bids[0]["price"]) if asks and bids else None,
    }


# ---------------------------------------------------------------------------
# Live Scanner logic
# ---------------------------------------------------------------------------

def scan_markets_with_books(hours_window, min_price, max_price, max_spend):
    now = datetime.now(timezone.utc)
    end_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_max = (now + timedelta(hours=hours_window)).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = fetch_all_markets(end_min, end_max)
    markets = [m for m in (parse_market(r) for r in raw) if m and m["clob_ids"]]
    # Filter out sports-related markets
    markets = [m for m in markets if not is_sports_market(m["question"], m.get("slug", ""))]
    all_rows, flagged_rows = [], []
    for m in markets:
        yes_p = m["prices"][0] if len(m["prices"]) >= 1 else None
        no_p = m["prices"][1] if len(m["prices"]) >= 2 else None
        td = m["end_date"] - now
        hours_left = max(td.total_seconds() / 3600, 0)
        expiry_str = f"{int(hours_left)}h {int((hours_left % 1) * 60)}m"
        yes_flag = yes_p is not None and min_price <= yes_p <= max_price
        no_flag = no_p is not None and min_price <= no_p <= max_price
        all_rows.append({"id": m["id"], "question": m["question"], "expiry_str": expiry_str,
                         "hours_left": round(hours_left, 2), "yes_price": yes_p, "no_price": no_p,
                         "volume": m["volume"], "liquidity": m["liquidity"], "flagged": yes_flag or no_flag})
        if not (yes_flag or no_flag):
            continue
        for i, (outcome, price) in enumerate(zip(m["outcomes"], m["prices"])):
            if i >= len(m["clob_ids"]) or not (min_price <= price <= max_price):
                continue
            try:
                ob = fetch_order_book(m["clob_ids"][i])
                time.sleep(REQUEST_DELAY)
                analysis = get_lift_analysis(ob, max_price=max_price, max_spend=max_spend)
            except Exception:
                analysis = None
            if analysis:
                verdict = "THIN"
                if analysis["range_depth"] >= 1000 and analysis["roi_pct"] >= 2.0:
                    verdict = "GOOD"
                elif analysis["range_depth"] >= 500 and analysis["roi_pct"] >= 1.0:
                    verdict = "OK"
                elif analysis["range_depth"] < 100:
                    verdict = "NO LIQ"
                flagged_rows.append({
                    "id": m["id"], "question": m["question"], "expiry": expiry_str, "side": outcome,
                    "mid_price": f"{price:.1%}",
                    "best_ask": f"{analysis['best_ask']:.3f}" if analysis["best_ask"] else "N/A",
                    "best_bid": f"{analysis['best_bid']:.3f}" if analysis["best_bid"] else "N/A",
                    "spread": f"{analysis['spread']:.3f}" if analysis["spread"] else "N/A",
                    "book_depth_shares": f"{analysis['range_depth']:,.0f}",
                    "book_depth_usd": f"${analysis['range_value']:,.0f}",
                    "max_buy_shares": f"{analysis['total_shares']:,.0f}",
                    "max_buy_cost": f"${analysis['total_cost']:,.0f}",
                    "avg_fill": f"{analysis['avg_fill']:.2%}", "worst_fill": f"{analysis['worst_fill']:.3f}",
                    "levels_eaten": str(analysis["levels"]),
                    "net_profit": f"${analysis['net_profit']:,.2f}", "roi": f"{analysis['roi_pct']:.2f}%",
                    "bid_depth_usd": f"${analysis['total_bid_value']:,.0f}", "verdict": verdict,
                    "_roi": analysis["roi_pct"], "_depth": analysis["range_depth"],
                    "_profit": analysis["net_profit"], "_clob_id": m["clob_ids"][i],
                    "_end_date": m["end_date"].isoformat(),
                })
            else:
                flagged_rows.append({
                    "id": m["id"], "question": m["question"], "expiry": expiry_str, "side": outcome,
                    "mid_price": f"{price:.1%}", "best_ask": "N/A", "best_bid": "N/A", "spread": "N/A",
                    "book_depth_shares": "0", "book_depth_usd": "$0", "max_buy_shares": "0",
                    "max_buy_cost": "$0", "avg_fill": "N/A", "worst_fill": "N/A", "levels_eaten": "0",
                    "net_profit": "$0", "roi": "0%", "bid_depth_usd": "$0", "verdict": "NO BOOK",
                    "_roi": 0, "_depth": 0, "_profit": 0,
                    "_clob_id": m["clob_ids"][i] if i < len(m["clob_ids"]) else "",
                    "_end_date": m["end_date"].isoformat(),
                })
    flagged_rows.sort(key=lambda r: r["_roi"], reverse=True)
    return all_rows, flagged_rows


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Polymarket Expiry Scanner", suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: #0d1117; color: #e6edf3; margin: 0; }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #161b22; }
            ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #484f58; }
            input[type=number] { color-scheme: dark; }
            .tab-container .tab {
                background-color: #161b22 !important;
                color: #8b949e !important;
                border: 1px solid #30363d !important;
                border-bottom: none !important;
                padding: 12px 24px !important;
                font-weight: bold !important;
            }
            .tab-container .tab--selected {
                background-color: #0d1117 !important;
                color: #58a6ff !important;
                border-bottom: 2px solid #58a6ff !important;
            }
            .dash-spinner { margin: 2rem auto; }
            ._dash-loading { position: relative; }
            ._dash-loading::after {
                content: "";
                position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(13,17,23,0.55);
                z-index: 999;
                pointer-events: none;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''


def make_stat_card(title, value, color=COLORS["text"]):
    return html.Div(style={
        "backgroundColor": COLORS["bg"], "borderRadius": "8px",
        "border": f"1px solid {COLORS['border']}", "padding": "12px 16px",
        "minWidth": "120px", "textAlign": "center",
    }, children=[
        html.Div(title, style={"color": COLORS["muted"], "fontSize": "11px", "marginBottom": "4px"}),
        html.Div(value, style={"color": color, "fontSize": "20px", "fontWeight": "bold"}),
    ])


INPUT_STYLE = {"backgroundColor": COLORS["card"], "color": COLORS["text"],
               "border": f"1px solid {COLORS['border']}", "padding": "8px", "borderRadius": "6px"}


# ==================== LAYOUT ====================

live_tab = html.Div([
    # Controls
    html.Div(style={"display": "flex", "gap": "15px", "marginBottom": "15px", "flexWrap": "wrap", "alignItems": "flex-end"}, children=[
        html.Div([html.Label("Hours", style={"color": COLORS["muted"], "fontSize": "12px"}),
                  dcc.Input(id="hours-input", type="number", value=2, min=1, max=168, style={**INPUT_STYLE, "width": "70px"})]),
        html.Div([html.Label("Min Price", style={"color": COLORS["muted"], "fontSize": "12px"}),
                  dcc.Input(id="min-price-input", type="number", value=0.92, min=0.5, max=0.99, step=0.01, style={**INPUT_STYLE, "width": "70px"})]),
        html.Div([html.Label("Max Price", style={"color": COLORS["muted"], "fontSize": "12px"}),
                  dcc.Input(id="max-price-input", type="number", value=0.99, min=0.5, max=1.0, step=0.01, style={**INPUT_STYLE, "width": "70px"})]),
        html.Div([html.Label("Max Spend", style={"color": COLORS["muted"], "fontSize": "12px"}),
                  dcc.Input(id="max-spend-input", type="number", value=1000, min=100, step=100, style={**INPUT_STYLE, "width": "90px"})]),
        html.Button("Scan Now", id="scan-btn", n_clicks=0, style={
            "backgroundColor": COLORS["blue"], "color": "#fff", "border": "none",
            "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
    ]),
    dcc.Loading(id="loading-scanner", type="circle", color=COLORS["blue"],
        overlay_style={"visibility": "visible"}, children=[
        html.Div(id="status-bar", children="Initializing scan...", style={"padding": "10px 15px", "backgroundColor": COLORS["card"],
            "borderRadius": "6px", "marginBottom": "15px", "border": f"1px solid {COLORS['border']}",
            "color": COLORS["muted"], "fontSize": "13px"}),
        html.Div(id="stats-cards", style={"display": "flex", "gap": "12px", "marginBottom": "15px", "flexWrap": "wrap"}),
        html.Div(id="optimal-trade-panel", style={"backgroundColor": COLORS["card"], "borderRadius": "8px",
            "border": f"2px solid {COLORS['green']}", "padding": "15px", "marginBottom": "15px"}),
        html.Div(style={"backgroundColor": COLORS["card"], "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}", "padding": "15px", "marginBottom": "15px"}, children=[
            html.H3("Flagged Markets with Order Book Analysis", style={"marginTop": "0", "color": COLORS["yellow"], "fontSize": "18px"}),
            html.P("Verdict: GOOD = deep book + high ROI | OK = tradeable | THIN = low depth | NO LIQ / NO BOOK",
                   style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "10px"}),
            html.Div(id="flagged-table-container"),
        ]),
        html.Div(style={"backgroundColor": COLORS["card"], "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}", "padding": "15px", "marginBottom": "15px"}, children=[
            html.H3("Order Book Detail", style={"marginTop": "0", "color": COLORS["purple"], "fontSize": "18px"}),
            html.P("Click a row above.", style={"color": COLORS["muted"], "fontSize": "12px"}),
            html.Div(id="orderbook-panel"),
        ]),
        html.Div(style={"backgroundColor": COLORS["card"], "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}", "padding": "15px", "marginBottom": "15px"}, children=[
            html.H3("All Near-Expiry Markets", style={"marginTop": "0", "color": COLORS["text"], "fontSize": "18px"}),
            html.Div(id="all-table-container"),
        ]),
    ]),
])


paper_tab = html.Div([
    # Controls
    html.Div(style={"display": "flex", "gap": "15px", "marginBottom": "15px", "alignItems": "flex-end", "flexWrap": "wrap"}, children=[
        html.Div([html.Label("Stop Loss %", style={"color": COLORS["muted"], "fontSize": "12px"}),
                  dcc.Input(id="pt-stop-loss-input", type="number", value=20, min=1, max=99, step=1, style={**INPUT_STYLE, "width": "70px"})]),
        html.Button("Scan Now", id="pt-scan-btn", n_clicks=0, style={
            "backgroundColor": COLORS["green"], "color": "#fff", "border": "none",
            "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
        html.Button("Refresh Positions", id="pt-refresh-btn", n_clicks=0, style={
            "backgroundColor": COLORS["blue"], "color": "#fff", "border": "none",
            "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
        html.Button("Reset Portfolio", id="pt-reset-btn", n_clicks=0, style={
            "backgroundColor": COLORS["red"], "color": "#fff", "border": "none",
            "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
    ]),
    # Status bar + content wrapped in Loading
    dcc.Loading(id="loading-paper", type="circle", color=COLORS["blue"],
        overlay_style={"visibility": "visible"}, children=[
        html.Div(id="pt-status", style={"padding": "10px 15px", "backgroundColor": COLORS["card"],
            "borderRadius": "6px", "marginBottom": "15px", "border": f"1px solid {COLORS['border']}",
            "color": COLORS["muted"], "fontSize": "13px"},
            children="Trades are placed automatically from the Live Scanner. GOOD and OK verdicts are auto-traded."),
        html.Div(id="pt-stats-cards", style={"display": "flex", "gap": "12px", "marginBottom": "15px", "flexWrap": "wrap"}),
        # Open Positions
        html.Div(style={"backgroundColor": COLORS["card"], "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}", "padding": "15px", "marginBottom": "15px"}, children=[
            html.H3("Open Positions", style={"marginTop": "0", "color": COLORS["yellow"], "fontSize": "18px"}),
            html.Div(id="pt-open-table"),
        ]),
        # Resolved Trades
        html.Div(style={"backgroundColor": COLORS["card"], "borderRadius": "8px",
            "border": f"1px solid {COLORS['border']}", "padding": "15px", "marginBottom": "15px"}, children=[
            html.H3("Resolved Trades", style={"marginTop": "0", "color": COLORS["purple"], "fontSize": "18px"}),
            html.Div(id="pt-resolved-table"),
        ]),
        # Charts row
        html.Div(style={"display": "flex", "gap": "15px", "marginBottom": "15px", "flexWrap": "wrap"}, children=[
            html.Div(style={"flex": "2", "minWidth": "400px", "backgroundColor": COLORS["card"], "borderRadius": "8px",
                "border": f"1px solid {COLORS['border']}", "padding": "15px"}, children=[
                html.H3("Equity Curve (Realized PnL)", style={"marginTop": "0", "color": COLORS["green"], "fontSize": "18px"}),
                dcc.Graph(id="pt-equity-chart", config={"displayModeBar": False}, style={"height": "300px", "width": "100%"}),
            ]),
            html.Div(style={"flex": "1", "minWidth": "250px", "backgroundColor": COLORS["card"], "borderRadius": "8px",
                "border": f"1px solid {COLORS['border']}", "padding": "15px"}, children=[
                html.H3("Win / Loss", style={"marginTop": "0", "color": COLORS["blue"], "fontSize": "18px"}),
                dcc.Graph(id="pt-winloss-chart", config={"displayModeBar": False}, style={"height": "300px", "width": "100%"}),
            ]),
        ]),
    ]),
])


app.layout = html.Div(style={
    "backgroundColor": COLORS["bg"], "color": COLORS["text"],
    "fontFamily": "'Segoe UI', -apple-system, sans-serif",
    "minHeight": "100vh", "padding": "20px", "maxWidth": "1600px", "margin": "0 auto",
}, children=[
    # Header with timer
    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "15px"}, children=[
        html.H1("Polymarket Near-Expiry Scanner", style={"color": COLORS["blue"], "marginBottom": "0", "fontSize": "28px"}),
        html.Div(id="timer-display", style={
            "color": COLORS["muted"], "fontSize": "13px", "backgroundColor": COLORS["card"],
            "padding": "6px 14px", "borderRadius": "6px", "border": f"1px solid {COLORS['border']}",
        }, children="No scan yet"),
    ]),
    dcc.Tabs(id="main-tabs", value="live", className="tab-container", children=[
        dcc.Tab(label="Live Scanner", value="live", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Paper Trading", value="paper", className="tab", selected_className="tab--selected"),
    ], style={"marginBottom": "15px"}),
    html.Div(id="tab-content"),
    html.Div(style={"textAlign": "center", "padding": "15px", "color": COLORS["muted"], "fontSize": "11px",
                     "borderTop": f"1px solid {COLORS['border']}"}, children=[
        html.P("RISK: Near-expiry arb carries resolution risk. Not financial advice."),
    ]),
    # Persistent stores (survive tab switches)
    dcc.Store(id="scan-data-store"), dcc.Store(id="flagged-data-store"),
    dcc.Store(id="pt-data-store"),
    dcc.Store(id="last-scan-ts"),
    dcc.Store(id="scan-trigger", data=0),
    dcc.Store(id="scan-params-store", data={"hours": 2, "min_p": 0.92, "max_p": 0.99, "max_spend": 1000}),
    dcc.Store(id="scan-status-store", data="Ready to scan..."),
    dcc.Interval(id="timer-tick", interval=1_000, disabled=False),
    dcc.Interval(id="auto-refresh-interval", interval=180_000, n_intervals=0, disabled=False),
    dcc.Interval(id="pt-refresh-interval", interval=180_000, disabled=False),
    dcc.Interval(id="initial-scan-trigger", interval=500, n_intervals=0, max_intervals=1, disabled=False),
])


# ==================== CALLBACKS ====================

app.clientside_callback(
    """
    function(n, ts) {
        if (!ts) return "No scan yet";
        var elapsed = Math.floor(Date.now() / 1000 - ts);
        if (elapsed < 0) elapsed = 0;
        var min = Math.floor(elapsed / 60);
        var sec = elapsed % 60;
        if (min > 0) return "Last scan: " + min + "m " + sec + "s ago";
        return "Last scan: " + sec + "s ago";
    }
    """,
    Output("timer-display", "children"),
    Input("timer-tick", "n_intervals"),
    State("last-scan-ts", "data"),
)


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "paper":
        return paper_tab
    return live_tab


# ----- Live Scanner Callbacks -----

# Trigger scan from button click
@app.callback(
    Output("scan-trigger", "data", allow_duplicate=True),
    Input("scan-btn", "n_clicks"),
    prevent_initial_call=True,
)
def trigger_scan_from_button(n_clicks):
    return (n_clicks or 0) + 1000  # offset to distinguish from interval


# Trigger scan from interval or initial load
@app.callback(
    Output("scan-trigger", "data", allow_duplicate=True),
    Input("auto-refresh-interval", "n_intervals"),
    Input("initial-scan-trigger", "n_intervals"),
    prevent_initial_call=True,
)
def trigger_scan_from_interval(n_intervals, initial_n):
    return (n_intervals or 0) + (initial_n or 0)


# Sync input values to persistent store
@app.callback(
    Output("scan-params-store", "data"),
    Input("hours-input", "value"), Input("min-price-input", "value"),
    Input("max-price-input", "value"), Input("max-spend-input", "value"),
    prevent_initial_call=True,
)
def sync_params(hours, min_p, max_p, max_spend):
    return {"hours": hours or 12, "min_p": min_p or 0.92, "max_p": max_p or 0.99, "max_spend": max_spend or 1000}


@app.callback(
    Output("scan-data-store", "data"), Output("flagged-data-store", "data"),
    Output("scan-status-store", "data"), Output("last-scan-ts", "data"),
    Input("scan-trigger", "data"),
    State("scan-params-store", "data"),
    prevent_initial_call=True,
)
def do_scan(scan_trigger, params):
    params = params or {}
    hours = params.get("hours", 2)
    min_p = params.get("min_p", 0.92)
    max_p = params.get("max_p", 0.99)
    max_spend = params.get("max_spend", 1000)
    now_epoch = time.time()
    try:
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Starting scan...", flush=True)
        all_rows, flagged_rows = scan_markets_with_books(hours, min_p, max_p, max_spend)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        good = sum(1 for r in flagged_rows if r["verdict"] == "GOOD")
        ok = sum(1 for r in flagged_rows if r["verdict"] == "OK")
        # Auto-place paper trades for best opportunities
        placed = auto_place_trades(flagged_rows)
        pt_msg = f" | {placed} new paper trade(s) placed" if placed else ""
        status = f"Last scan: {now_str} | {len(all_rows)} markets | {len(flagged_rows)} flagged ({good} GOOD, {ok} OK){pt_msg} | Auto-refresh 3min"
        print(f"[{now_str}] Scan complete: {len(all_rows)} markets, {len(flagged_rows)} flagged ({good} GOOD, {ok} OK){pt_msg}", flush=True)
        return all_rows, flagged_rows, status, now_epoch
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Scan failed: {e}", flush=True)
        return [], [], f"Scan failed: {e}", now_epoch


# Sync status store to status bar (when it exists in the rendered tab)
@app.callback(
    Output("status-bar", "children"),
    Input("scan-status-store", "data"),
    prevent_initial_call=True,
)
def sync_status_bar(status):
    return status or "Ready to scan..."


@app.callback(
    Output("stats-cards", "children"), Output("optimal-trade-panel", "children"),
    Output("flagged-table-container", "children"), Output("all-table-container", "children"),
    Input("scan-data-store", "data"), Input("flagged-data-store", "data"),
)
def update_visuals(all_data, flagged_data):
    if not all_data:
        empty = html.P("Scanning...", style={"color": COLORS["muted"]})
        return [], empty, "Scanning...", "Scanning..."
    df = pd.DataFrame(all_data)
    n_flagged = len(flagged_data) if flagged_data else 0
    good = sum(1 for r in (flagged_data or []) if r["verdict"] == "GOOD")
    ok = sum(1 for r in (flagged_data or []) if r["verdict"] == "OK")
    total_potential = sum(r["_profit"] for r in (flagged_data or []))
    cards = [
        make_stat_card("Total", str(len(df)), COLORS["blue"]),
        make_stat_card("Flagged", str(n_flagged), COLORS["yellow"]),
        make_stat_card("GOOD", str(good), COLORS["green"]),
        make_stat_card("OK", str(ok), COLORS["blue"]),
        make_stat_card("Avg Expiry", f"{df['hours_left'].mean():.1f}h", COLORS["text"]),
        make_stat_card("Sum Profit", f"${total_potential:,.0f}", COLORS["green"]),
    ]

    # Flagged table
    if flagged_data:
        flagged_table = dash_table.DataTable(
            id="flagged-table",
            columns=[{"name": n, "id": i} for n, i in [
                ("Verdict","verdict"),("Market","question"),("Expiry","expiry"),("Side","side"),
                ("Price","mid_price"),("Bid","best_bid"),("Ask","best_ask"),("Spread","spread"),
                ("Depth","book_depth_shares"),("Depth $","book_depth_usd"),
                ("Can Buy","max_buy_shares"),("Cost","max_buy_cost"),("Avg Fill","avg_fill"),
                ("Worst Fill","worst_fill"),("Lvls","levels_eaten"),
                ("Profit","net_profit"),("ROI","roi"),("Bid $","bid_depth_usd"),
            ]],
            data=[{k: v for k, v in r.items() if not k.startswith("_")} for r in flagged_data],
            row_selectable="single", style_table={"overflowX": "auto"},
            style_header=HEADER_STYLE,
            style_cell={**CELL_STYLE, "maxWidth": "200px", "overflow": "hidden", "textOverflow": "ellipsis"},
            style_cell_conditional=[{"if": {"column_id": "question"}, "maxWidth": "400px", "minWidth": "200px", "whiteSpace": "normal"}],
            style_data_conditional=[
                {"if": {"state": "selected"}, "backgroundColor": COLORS["border"], "border": f"1px solid {COLORS['yellow']}"},
                {"if": {"filter_query": '{verdict} = "GOOD"', "column_id": "verdict"}, "backgroundColor": "#1a3a1a", "color": COLORS["green"], "fontWeight": "bold"},
                {"if": {"filter_query": '{verdict} = "OK"', "column_id": "verdict"}, "backgroundColor": "#1a2a3a", "color": COLORS["blue"], "fontWeight": "bold"},
                {"if": {"filter_query": '{verdict} = "THIN"', "column_id": "verdict"}, "backgroundColor": "#3a2a1a", "color": COLORS["yellow"]},
                {"if": {"filter_query": '{verdict} = "NO LIQ"', "column_id": "verdict"}, "backgroundColor": "#3a1a1a", "color": COLORS["red"]},
                {"if": {"filter_query": '{verdict} = "NO BOOK"', "column_id": "verdict"}, "backgroundColor": "#2a1a1a", "color": COLORS["muted"]},
            ],
            page_size=25, sort_action="native",
        )
    else:
        flagged_table = html.P("No markets in target range.", style={"color": COLORS["muted"]})

    # All markets table
    adf = df[["id","question","expiry_str","yes_price","no_price","volume","liquidity","flagged"]].copy()
    adf["yes_price"] = adf["yes_price"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    adf["no_price"] = adf["no_price"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    adf["volume"] = adf["volume"].apply(lambda x: f"${x:,.0f}")
    adf["liquidity"] = adf["liquidity"].apply(lambda x: f"${x:,.0f}")
    adf["flagged"] = adf["flagged"].apply(lambda x: "YES" if x else "")
    all_table = dash_table.DataTable(
        id="all-markets-table",
        columns=[{"name": n, "id": i} for n, i in [("ID","id"),("Market","question"),("Expiry","expiry_str"),
                 ("YES","yes_price"),("NO","no_price"),("Volume","volume"),("Liq","liquidity"),("Flag","flagged")]],
        data=adf.to_dict("records"), style_table={"overflowX": "auto"},
        style_header=HEADER_STYLE,
        style_cell={**CELL_STYLE, "maxWidth": "300px", "overflow": "hidden", "textOverflow": "ellipsis"},
        page_size=20, sort_action="native", filter_action="native",
    )

    # Optimal trade
    optimal_panel = html.P("No tradeable opportunities.", style={"color": COLORS["muted"]})
    if flagged_data:
        tradeable = [r for r in flagged_data if r["_roi"] > 0 and r["_depth"] > 50]
        if tradeable:
            for r in tradeable:
                r["_score"] = r["_roi"] * 0.4 + min(r["_depth"]/1000, 1.0) * 100 * 0.3 + min(r["_profit"]/100, 1.0) * 100 * 0.3
            tradeable.sort(key=lambda r: r["_score"], reverse=True)
            best = tradeable[0]
            best_section = html.Div(style={"backgroundColor": "#0d2818", "borderRadius": "8px", "padding": "15px",
                "marginBottom": "10px", "border": f"1px solid {COLORS['green']}"}, children=[
                html.Div("BEST TRADE", style={"color": COLORS["green"], "fontWeight": "bold", "fontSize": "11px", "letterSpacing": "1px", "marginBottom": "8px"}),
                html.Div(best['question'], style={"color": COLORS["text"], "fontSize": "16px", "fontWeight": "bold", "marginBottom": "5px"}),
                html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}, children=[
                    html.Span(f"Side: {best['side']}", style={"color": COLORS["yellow"]}),
                    html.Span(f"Price: {best['mid_price']}", style={"color": COLORS["text"]}),
                    html.Span(f"Expiry: {best['expiry']}", style={"color": COLORS["muted"]}),
                    html.Span(f"Buy {best['max_buy_shares']} @ {best['avg_fill']}", style={"color": COLORS["blue"]}),
                    html.Span(f"Cost: {best['max_buy_cost']}", style={"color": COLORS["text"]}),
                    html.Span(f"Profit: {best['net_profit']}", style={"color": COLORS["green"], "fontWeight": "bold"}),
                    html.Span(f"ROI: {best['roi']}", style={"color": COLORS["green"], "fontWeight": "bold"}),
                    html.Span(f"Depth: {best['book_depth_shares']} ({best['book_depth_usd']})", style={"color": COLORS["purple"]}),
                ]),
            ])
            runners = [html.Div(style={"backgroundColor": COLORS["bg"], "borderRadius": "6px", "padding": "10px",
                "border": f"1px solid {COLORS['border']}", "flex": "1", "minWidth": "300px"}, children=[
                html.Div(f"#{i}", style={"color": COLORS["muted"], "fontSize": "11px"}),
                html.Div(r["question"], style={"color": COLORS["text"], "fontSize": "13px", "fontWeight": "bold"}),
                html.Div(f"{r['side']} @ {r['mid_price']} | ROI: {r['roi']} | Profit: {r['net_profit']} | Depth: {r['book_depth_shares']}",
                         style={"color": COLORS["muted"], "fontSize": "12px", "marginTop": "4px"}),
            ]) for i, r in enumerate(tradeable[1:3], 2)]
            optimal_panel = html.Div([
                html.H3("Optimal Trade", style={"marginTop": "0", "color": COLORS["green"], "fontSize": "18px"}),
                html.P("Score = ROI(40%) + Depth(30%) + Profit(30%)", style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "10px"}),
                best_section,
                html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=runners) if runners else html.Div(),
            ])

    return cards, optimal_panel, flagged_table, all_table


@app.callback(
    Output("orderbook-panel", "children"),
    Input("flagged-table", "selected_rows"),
    State("flagged-data-store", "data"), State("max-price-input", "value"), State("max-spend-input", "value"),
    prevent_initial_call=True,
)
def show_order_book(selected_rows, flagged_data, max_p, max_spend):
    if not selected_rows or not flagged_data:
        return html.P("Select a market above.", style={"color": COLORS["muted"]})
    max_p, max_spend = max_p or 0.99, max_spend or 10000
    idx = selected_rows[0]
    if idx >= len(flagged_data):
        return html.P("Invalid.", style={"color": COLORS["red"]})
    market = flagged_data[idx]
    clob_id = market.get("_clob_id", "")
    if not clob_id:
        return html.P("No token ID.", style={"color": COLORS["red"]})
    panels = [
        html.H4(f"{market['question']} - {market['side']}", style={"color": COLORS["yellow"], "marginBottom": "5px"}),
        html.P(f"Expiry: {market['expiry']} | Price: {market['mid_price']} | Verdict: {market['verdict']} | ROI: {market['roi']}",
               style={"color": COLORS["muted"], "fontSize": "13px"}),
    ]
    try:
        ob = fetch_order_book(clob_id)
    except Exception as e:
        panels.append(html.P(f"Failed: {e}", style={"color": COLORS["red"]}))
        return html.Div(panels)
    bids = sorted([{"price": float(b["price"]), "size": float(b["size"])} for b in ob.get("bids", [])], key=lambda x: x["price"], reverse=True)
    asks = sorted([{"price": float(a["price"]), "size": float(a["size"])} for a in ob.get("asks", [])], key=lambda x: x["price"])
    fig = go.Figure()
    if bids:
        cum, r = [], 0
        for b in bids: r += b["size"]; cum.append({"p": b["price"], "c": r})
        fig.add_trace(go.Scatter(x=[c["p"] for c in cum], y=[c["c"] for c in cum], fill="tozeroy", name="Bids",
                                 line=dict(color=COLORS["green"]), fillcolor="rgba(63,185,80,0.2)"))
    if asks:
        cum, r = [], 0
        for a in asks: r += a["size"]; cum.append({"p": a["price"], "c": r})
        fig.add_trace(go.Scatter(x=[c["p"] for c in cum], y=[c["c"] for c in cum], fill="tozeroy", name="Asks",
                                 line=dict(color=COLORS["red"]), fillcolor="rgba(248,81,73,0.2)"))
    fig.update_layout(title=f"{market['side']} Depth", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["bg"],
                      font_color=COLORS["text"], margin=dict(l=50,r=20,t=40,b=40),
                      xaxis=dict(title="Price", tickformat=".3f", gridcolor=COLORS["border"]),
                      yaxis=dict(title="Cumulative Size", gridcolor=COLORS["border"]),
                      height=350, autosize=False, legend=dict(bgcolor="rgba(0,0,0,0)"))
    panels.append(dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "350px", "width": "100%"}))
    analysis = get_lift_analysis(ob, max_price=max_p, max_spend=max_spend)
    if analysis:
        panels.append(html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "15px 0"}, children=[
            make_stat_card("Avg Fill", f"{analysis['avg_fill']:.2%}", COLORS["blue"]),
            make_stat_card("Cost", f"${analysis['total_cost']:,.0f}", COLORS["text"]),
            make_stat_card("Shares", f"${analysis['total_shares']:,.0f}", COLORS["text"]),
            make_stat_card("Profit", f"${analysis['net_profit']:,.2f}", COLORS["green"]),
            make_stat_card("ROI", f"{analysis['roi_pct']:.2f}%", COLORS["green"] if analysis["roi_pct"] > 0 else COLORS["red"]),
            make_stat_card("Levels", str(analysis["levels"]), COLORS["purple"]),
        ]))
        fill_rows = [{"price": f"{f['price']:.4f}", "size": f"{f['size']:,.2f}", "cost": f"${f['price']*f['size']:,.2f}"} for f in analysis["fills"][:20]]
        panels.append(dash_table.DataTable(columns=[{"name":"Price","id":"price"},{"name":"Size","id":"size"},{"name":"Cost","id":"cost"}],
                                           data=fill_rows, style_header=HEADER_STYLE, style_cell=CELL_STYLE))
    return html.Div(panels)


# ----- Paper Trading Callbacks -----

@app.callback(
    Output("pt-data-store", "data"), Output("pt-status", "children"),
    Input("pt-scan-btn", "n_clicks"), Input("pt-refresh-btn", "n_clicks"),
    Input("pt-reset-btn", "n_clicks"), Input("pt-refresh-interval", "n_intervals"),
    State("pt-stop-loss-input", "value"),
    prevent_initial_call=True,
)
def handle_paper_trade_actions(scan_clicks, refresh_clicks, reset_clicks, n_intervals, stop_loss_val):
    stop_loss_pct = (stop_loss_val or 20) / 100.0
    ctx = dash.callback_context
    if not ctx.triggered:
        trades = check_and_resolve_trades(stop_loss_pct)
        return trades, "Refreshed."
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "pt-reset-btn":
        save_paper_trades([])
        return [], "Portfolio reset. All trades cleared."

    if trigger == "pt-scan-btn":
        try:
            _, flagged_rows = scan_markets_with_books(2, 0.92, 0.99, 1000)
            placed = auto_place_trades(flagged_rows)
            trades = check_and_resolve_trades(stop_loss_pct)
            now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            good = sum(1 for r in flagged_rows if r["verdict"] == "GOOD")
            n_open = sum(1 for t in trades if t["status"] == "open")
            n_resolved = sum(1 for t in trades if t["status"].startswith("resolved"))
            pt_msg = f" | {placed} new trade(s) placed" if placed else " | no new trades"
            return trades, f"Scan at {now_str} | {len(flagged_rows)} flagged ({good} GOOD){pt_msg} | {n_open} open, {n_resolved} resolved"
        except Exception as e:
            trades = load_paper_trades()
            return trades, f"Scan failed: {e}"

    # refresh or interval
    trades = check_and_resolve_trades(stop_loss_pct)
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    n_open = sum(1 for t in trades if t["status"] == "open")
    n_resolved = sum(1 for t in trades if t["status"].startswith("resolved"))
    n_stopped = sum(1 for t in trades if "Stop Loss" in (t.get("resolution") or ""))
    sl_msg = f" | {n_stopped} stopped out" if n_stopped else ""
    return trades, f"Refreshed at {now_str} | {n_open} open, {n_resolved} resolved{sl_msg} | SL: {stop_loss_pct:.0%}"


@app.callback(
    Output("pt-stats-cards", "children"), Output("pt-open-table", "children"),
    Output("pt-resolved-table", "children"), Output("pt-equity-chart", "figure"),
    Output("pt-winloss-chart", "figure"),
    Input("pt-data-store", "data"),
    prevent_initial_call=True,
)
def update_paper_visuals(trades):
    empty_fig = go.Figure()
    empty_fig.update_layout(paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
                            font_color=COLORS["muted"], height=280, autosize=False,
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            annotations=[dict(text="No data", showarrow=False, font=dict(size=16, color=COLORS["muted"]))])

    if not trades:
        return [], "No trades yet.", "No resolved trades.", empty_fig, empty_fig

    open_trades = [t for t in trades if t["status"] == "open"]
    resolved_trades = [t for t in trades if t["status"].startswith("resolved")]
    n_total = len(trades)
    n_open = len(open_trades)
    n_resolved = len(resolved_trades)
    n_wins = sum(1 for t in resolved_trades if t["status"] == "resolved_win")
    n_losses = sum(1 for t in resolved_trades if t["status"] == "resolved_loss")
    win_rate = (n_wins / n_resolved * 100) if n_resolved > 0 else 0
    realized_pnl = sum(t.get("net_pnl", 0) or 0 for t in resolved_trades)
    total_invested = sum(t.get("cost", 0) for t in open_trades)

    # Compute unrealized PnL by fetching current prices
    unrealized_pnl = 0.0
    current_prices = {}
    if open_trades:
        try:
            current_prices = get_current_prices(open_trades)
        except Exception:
            pass
        for t in open_trades:
            cp = current_prices.get(t["trade_id"])
            if cp is not None:
                unrealized_pnl += (cp - t["entry_price"]) * t["shares"]

    # Sum estimated profit from book snapshots of open trades
    est_profit = 0.0
    for t in open_trades:
        bs = t.get("book_snapshot", {})
        try:
            val = bs.get("net_profit_est", "$0").replace("$", "").replace(",", "")
            est_profit += float(val)
        except (ValueError, TypeError):
            pass

    cards = [
        make_stat_card("Total Trades", str(n_total), COLORS["blue"]),
        make_stat_card("Open", str(n_open), COLORS["yellow"]),
        make_stat_card("Resolved", str(n_resolved), COLORS["purple"]),
        make_stat_card("Win Rate", f"{win_rate:.0f}%", COLORS["green"] if win_rate > 50 else COLORS["red"]),
        make_stat_card("Realized PnL", f"${realized_pnl:,.2f}", COLORS["green"] if realized_pnl >= 0 else COLORS["red"]),
        make_stat_card("Unrealized PnL", f"${unrealized_pnl:,.2f}", COLORS["green"] if unrealized_pnl >= 0 else COLORS["red"]),
        make_stat_card("Est Profit", f"${est_profit:,.2f}", COLORS["green"] if est_profit >= 0 else COLORS["red"]),
        make_stat_card("Invested", f"${total_invested:,.2f}", COLORS["text"]),
    ]

    # Open Positions table with book snapshot
    if open_trades:
        open_rows = []
        upnl_nums = []
        for t in open_trades:
            cp = current_prices.get(t["trade_id"])
            upnl = (cp - t["entry_price"]) * t["shares"] if cp is not None else None
            upnl_nums.append(upnl)
            time_left = ""
            if t.get("expiry_time"):
                try:
                    exp = datetime.fromisoformat(t["expiry_time"].replace("Z", "+00:00"))
                    td = exp - datetime.now(timezone.utc)
                    if td.total_seconds() > 0:
                        hrs = int(td.total_seconds() // 3600)
                        mins = int((td.total_seconds() % 3600) // 60)
                        time_left = f"{hrs}h {mins}m"
                    else:
                        time_left = "Expired"
                except Exception:
                    time_left = t.get("expiry_time", "")
            bs = t.get("book_snapshot", {})
            open_rows.append({
                "trade_id": t["trade_id"],
                "market": t["question"] or t["market_id"],
                "side": t["side"],
                "entry": f"{t['entry_price']:.4f}",
                "shares": f"{t['shares']:,.1f}",
                "cost": f"${t['cost']:,.2f}",
                "current": f"{cp:.4f}" if cp is not None else "...",
                "upnl": f"${upnl:,.2f}" if upnl is not None else "...",
                "time_left": time_left,
                "verdict": bs.get("verdict", ""),
                "bid": bs.get("best_bid", ""),
                "ask": bs.get("best_ask", ""),
                "spread": bs.get("spread", ""),
                "depth": bs.get("book_depth_shares", ""),
                "depth_usd": bs.get("book_depth_usd", ""),
                "est_roi": bs.get("roi_at_entry", ""),
                "est_profit": bs.get("net_profit_est", ""),
            })
        # Heatmap styles for Current PnL column
        pnl_styles = []
        valid_abs = [abs(v) for v in upnl_nums if v is not None]
        max_abs = max(valid_abs) if valid_abs else 1.0
        max_abs = max_abs or 1.0  # avoid div-by-zero
        for i, val in enumerate(upnl_nums):
            if val is None:
                continue
            intensity = min(abs(val) / max_abs, 1.0)
            if val >= 0:
                bg = f"rgba(63, 185, 80, {0.1 + intensity * 0.55:.2f})"
                fg = COLORS["green"]
            else:
                bg = f"rgba(248, 81, 73, {0.1 + intensity * 0.55:.2f})"
                fg = COLORS["red"]
            pnl_styles.append({
                "if": {"row_index": i, "column_id": "upnl"},
                "backgroundColor": bg, "color": fg, "fontWeight": "bold",
            })
        open_table = dash_table.DataTable(
            columns=[{"name": n, "id": i} for n, i in [
                ("ID", "trade_id"), ("Market", "market"), ("Side", "side"), ("Entry", "entry"),
                ("Shares", "shares"), ("Cost", "cost"), ("Current", "current"),
                ("Current PnL", "upnl"), ("Time Left", "time_left"),
                ("Verdict", "verdict"), ("Bid", "bid"), ("Ask", "ask"), ("Spread", "spread"),
                ("Depth", "depth"), ("Depth $", "depth_usd"),
                ("Est ROI", "est_roi"), ("Est Profit", "est_profit")]],
            data=open_rows, style_table={"overflowX": "auto"}, style_header=HEADER_STYLE,
            style_cell={**CELL_STYLE, "maxWidth": "180px", "overflow": "hidden", "textOverflow": "ellipsis"},
            style_cell_conditional=[{"if": {"column_id": "market"}, "maxWidth": "400px", "minWidth": "200px", "whiteSpace": "normal"}],
            style_data_conditional=[
                {"if": {"filter_query": '{verdict} = "GOOD"', "column_id": "verdict"}, "backgroundColor": "#1a3a1a", "color": COLORS["green"], "fontWeight": "bold"},
                {"if": {"filter_query": '{verdict} = "OK"', "column_id": "verdict"}, "backgroundColor": "#1a2a3a", "color": COLORS["blue"], "fontWeight": "bold"},
            ] + pnl_styles,
            page_size=20, sort_action="native",
        )
    else:
        open_table = html.P("No open positions.", style={"color": COLORS["muted"]})

    # Resolved Trades table with book snapshot and red-flag losses
    if resolved_trades:
        resolved_rows = []
        for t in resolved_trades:
            roi = ((t.get("net_pnl", 0) or 0) / t["cost"] * 100) if t.get("cost", 0) > 0 else 0
            result = "WIN" if t["status"] == "resolved_win" else "LOSS"
            bs = t.get("book_snapshot", {})
            resolved_rows.append({
                "trade_id": t["trade_id"],
                "market": t["question"] or t["market_id"],
                "side": t["side"],
                "entry": f"{t['entry_price']:.4f}",
                "shares": f"{t['shares']:,.1f}",
                "cost": f"${t['cost']:,.2f}",
                "resolution": t.get("resolution", ""),
                "result": result,
                "pnl": f"${(t.get('net_pnl', 0) or 0):,.2f}",
                "roi": f"{roi:.1f}%",
                "resolved_at": (t.get("resolved_at", "") or "")[:16],
                "verdict": bs.get("verdict", ""),
                "bid": bs.get("best_bid", ""),
                "ask": bs.get("best_ask", ""),
                "spread": bs.get("spread", ""),
                "depth": bs.get("book_depth_shares", ""),
                "depth_usd": bs.get("book_depth_usd", ""),
                "est_roi": bs.get("roi_at_entry", ""),
                "est_profit": bs.get("net_profit_est", ""),
            })
        resolved_table = dash_table.DataTable(
            columns=[{"name": n, "id": i} for n, i in [
                ("ID", "trade_id"), ("Market", "market"), ("Side", "side"), ("Entry", "entry"),
                ("Shares", "shares"), ("Cost", "cost"), ("Resolution", "resolution"),
                ("Result", "result"), ("PnL", "pnl"), ("ROI", "roi"), ("Resolved", "resolved_at"),
                ("Verdict", "verdict"), ("Bid@Entry", "bid"), ("Ask@Entry", "ask"), ("Spread@Entry", "spread"),
                ("Depth@Entry", "depth"), ("Depth$@Entry", "depth_usd"),
                ("Est ROI", "est_roi"), ("Est Profit", "est_profit")]],
            data=resolved_rows, style_table={"overflowX": "auto"}, style_header=HEADER_STYLE,
            style_cell={**CELL_STYLE, "maxWidth": "180px", "overflow": "hidden", "textOverflow": "ellipsis"},
            style_cell_conditional=[{"if": {"column_id": "market"}, "maxWidth": "400px", "minWidth": "200px", "whiteSpace": "normal"}],
            style_data_conditional=[
                {"if": {"filter_query": '{result} = "WIN"', "column_id": "result"}, "color": COLORS["green"], "fontWeight": "bold"},
                {"if": {"filter_query": '{result} = "LOSS"', "column_id": "result"}, "color": COLORS["red"], "fontWeight": "bold"},
                # Red-flag entire row for losses
                {"if": {"filter_query": '{result} = "LOSS"'}, "backgroundColor": "#2a1015", "border": f"1px solid {COLORS['red']}"},
                {"if": {"filter_query": '{result} = "WIN"'}, "backgroundColor": "#0d2818"},
            ],
            page_size=25, sort_action="native",
        )
    else:
        resolved_table = html.P("No resolved trades yet.", style={"color": COLORS["muted"]})

    # Equity curve (cumulative realized PnL over resolved trades)
    fig_equity = go.Figure()
    if resolved_trades:
        sorted_resolved = sorted(resolved_trades, key=lambda t: t.get("resolved_at", "") or "")
        cum_pnl = []
        running = 0
        for t in sorted_resolved:
            running += t.get("net_pnl", 0) or 0
            cum_pnl.append(running)
        fig_equity.add_trace(go.Scatter(
            x=list(range(1, len(cum_pnl) + 1)), y=cum_pnl,
            fill="tozeroy", name="Realized PnL",
            line=dict(color=COLORS["green"] if running >= 0 else COLORS["red"], width=2),
            fillcolor="rgba(63,185,80,0.15)" if running >= 0 else "rgba(248,81,73,0.15)",
        ))
        fig_equity.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    fig_equity.update_layout(
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
        margin=dict(l=50, r=20, t=20, b=40), height=280, autosize=False,
        xaxis=dict(title="Trade #", gridcolor=COLORS["border"]),
        yaxis=dict(title="Cumulative PnL ($)", gridcolor=COLORS["border"]),
        showlegend=False,
    )

    # Win/Loss bar chart
    fig_wl = go.Figure()
    fig_wl.add_trace(go.Bar(
        x=["Wins", "Losses"], y=[n_wins, n_losses],
        marker_color=[COLORS["green"], COLORS["red"]],
        text=[str(n_wins), str(n_losses)], textposition="auto",
        textfont=dict(color=COLORS["text"], size=16),
    ))
    fig_wl.update_layout(
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
        margin=dict(l=40, r=20, t=20, b=40), height=280, autosize=False,
        xaxis=dict(gridcolor=COLORS["border"]),
        yaxis=dict(title="Count", gridcolor=COLORS["border"]),
        showlegend=False,
    )

    return cards, open_table, resolved_table, fig_equity, fig_wl


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket Expiry Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    args = parser.parse_args()
    print(f"\nPolymarket Near-Expiry Dashboard")
    print(f"  Open http://127.0.0.1:{args.port}")
    print(f"  Tabs: Live Scanner | Paper Trading")
    print(f"  Press Ctrl+C to stop.\n")
    app.run(debug=not args.no_debug, port=args.port, host="127.0.0.1")


if __name__ == "__main__":
    main()
