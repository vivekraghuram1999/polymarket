# Polymarket Project Memory

## Project Structure
- `C:\Users\Ary\Downloads\polymarket\scanner.py` - Live near-expiry arbitrage scanner
- `C:\Users\Ary\Downloads\polymarket\data_pull.py` - Empty placeholder file

## Polymarket API Details (Verified Feb 2026)

### Gamma Markets API
- Base: `https://gamma-api.polymarket.com/markets`
- Key params: `limit`, `offset`, `active`, `closed`, `end_date_min`, `end_date_max`, `order`, `ascending`
- Max page size: 100, pagination via `offset`
- Key fields: `id`, `question`, `slug`, `endDate` (ISO), `outcomes` (array), `outcomePrices` (array of strings), `clobTokenIds` (array of long numeric strings), `bestBid`, `bestAsk`, `spread`, `volume`, `volumeNum`, `liquidity`, `liquidityNum`, `lastTradePrice`, `negRisk`, `active`, `closed`
- `outcomePrices` and `clobTokenIds` may be JSON strings or arrays depending on context
- Market categories include sports, crypto up/down, esports, entertainment props

### CLOB API
- Base: `https://clob.polymarket.com`
- Order book: `GET /book?token_id={id}` - returns `bids`/`asks` arrays with `price`/`size` fields
- Price: `GET /price?token_id={id}&side=BUY|SELL`
- Midpoint: `GET /midpoint?token_id={id}`
- Spreads: `POST /spreads` with array of `{token_id}` objects (max 500)
- Book response also includes: `market`, `asset_id`, `timestamp`, `hash`, `min_order_size`, `tick_size`, `neg_risk`, `last_trade_price`
- `tick_size` typically "0.001", `min_order_size` typically "5"

### Fee Structure
- ~2% on profits (payout - cost), not on full notional

## Windows Environment Notes
- Console encoding: Must use `io.TextIOWrapper` with `utf-8` + `errors="replace"` for market names with unicode characters (e.g., Portuguese team names)
- Python 3.12 installed via Microsoft Store

## Scanner Design Decisions
- Strategy: buy high-probability outcomes (92-99c) near expiry, collect $1 at resolution
- Lift analysis simulates walking up the ask side of the order book
- Configurable via CLI args: `--hours`, `--min-price`, `--max-price`, `--max-spend`, `--refresh`
- Rate limiting: 0.15s delay between CLOB API calls
- Handles multi-outcome markets (not just Yes/No - e.g., "Up"/"Down", team names)
