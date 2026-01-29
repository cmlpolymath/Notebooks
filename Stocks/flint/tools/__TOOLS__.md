# TOOLS.MD

## Table of Contents

- [ticker_tape.py](#ticker_tapepy)
- [tick.py](#tickpy)
---

## ticker_tape.py

`ticker_tape` is a **command-line tool for building a realistic, modeling-ready U.S. equity ticker universe**.
It produces a **deduplicated, validated set of common-stock tickers** spanning:

* Major U.S. listed exchanges (NYSE, Nasdaq, NYSE Arca, etc.)
* **OTC equities** via FINRA CAT (authoritative, reportable OTC symbols)

The output is written to **Parquet**, ready for downstream OHLCV scraping and quantitative modeling.

---

### What this tool does

At a high level:

1. **Discovers tickers** from authoritative public sources:

   * NasdaqTrader symbol directories (NYSE, Nasdaq, Arca, etc.)
   * FINRA CAT Reportable Equity Symbol Master (OTC)
2. **Normalizes and validates** symbols:

   * Uppercases tickers
   * Removes test issues
   * Excludes ETFs
   * Applies conservative “common stock” symbol rules
3. **Unifies and deduplicates**:

   * Merges listed + OTC universes
   * Prefers listed tickers when duplicates exist
4. **Annotates metadata** useful for modeling:

   * `ticker_type` → `listed` vs `otc`
   * `exchange` / `exchange_name`
   * `source` (NasdaqTrader vs FINRA CAT)
5. **Exports Parquet** for fast, columnar downstream processing

The result is a **market-realistic equity universe**, not just “whatever Yahoo shows”.

---

### Why this exists (design intent)

This tool is intentionally **not** a scraper and **not** tied to a broker API.

It exists to give you:

* A **stable, auditable universe definition**
* Separation between **universe construction** and **price data ingestion**
* The ability to **stratify models** (listed vs OTC behave very differently)
* High performance and low memory overhead for large symbol sets

---

### Key features

#### Async I/O

* Concurrent HTTP downloads using `asyncio` + `httpx`
* Bounded concurrency for any blocking calls (e.g. yfinance sampling)

#### Polars (not pandas)

* Fast CSV parsing (Rust engine)
* Low memory footprint
* Efficient deduplication and grouping
* Native Parquet writer with statistics

#### Validation & sanity checks

* Cross-checks listed tickers against SEC exchange metadata
* Filters test issues and malformed symbols
* Explicit exchange grouping (no silent assumptions)

#### Clean separation of concerns

* **Universe building** → `common` command
* **Inspection** → `stats`
* **Sampling / smoke tests** → `sample-info`
* OHLCV scraping happens *after* this step, not inside it

---

### Installation

```bash
pip install typer rich polars httpx yfinance
# or
uv add typer rich polars httpx yfinance
```

---

### Primary command (the one you’ll actually run)

#### Build the full modeling universe (listed + OTC)

```bash
python ticker_tape.py common \
  --out data_drops/universe_common.parquet
```

This single command:

* Builds the **union of all major U.S. listed equities + OTC**
* Deduplicates symbols
* Labels each row with `ticker_type` and `exchange`
* Writes a Parquet file you can immediately feed into OHLCV scraping

---

### Output schema (Parquet)

| column          | description                         |
| --------------- | ----------------------------------- |
| `ticker`        | Uppercase equity ticker             |
| `name`          | Security name                       |
| `ticker_type`   | `listed` or `otc`                   |
| `security_type` | `common`                            |
| `exchange`      | Exchange code (`N`, `Q`, `U`, etc.) |
| `exchange_name` | Human-readable exchange             |
| `source`        | `nasdaqtrader` or `finra_cat`       |

---

### Inspecting and validating the universe

#### Show counts by exchange and ticker type

```bash
python ticker_tape.py stats data_drops/universe_common.parquet
```

#### Example Python sanity check

```python
import polars as pl

df = pl.read_parquet("data_drops/universe_common.parquet")
print(df.group_by(["ticker_type", "exchange"]).len())
```

---

### Sampling fundamentals (optional sanity check)

This does **not** build the universe again — it samples from the saved Parquet.

```bash
python ticker_tape.py sample-info \
  --inp data_drops/universe_common.parquet \
  --n 10 \
  --max-concurrency 8
```

Useful for:

* Verifying symbols resolve in Yahoo Finance
* Spot-checking coverage before long OHLCV jobs

---

### Typical workflow (end-to-end)

```bash
# 1. Build universe
python ticker_tape.py common --out data_drops/universe_common.parquet

# 2. Inspect counts
python ticker_tape.py stats data_drops/universe_common.parquet

# 3. (Optional) Sample fundamentals
python ticker_tape.py sample-info --inp data_drops/universe_common.parquet
```

---

## tick.py

A production-grade Python script for downloading historical stock market data (OHLCV) and fundamental financial statements from free data sources.

### Features

- Downloads 10 years of daily OHLCV data using Yahoo Finance (yfinance)
- Retrieves fundamental statements (income, balance sheet, cash flow) via Alpha Vantage
- Fallback support for Robinhood data source
- DuckDB-based state management for idempotent execution
- Automatic retry logic with exponential backoff
- Rate limiting to avoid API throttling
- Parquet file storage for efficient data access

### Requirements

```bash
pip install pandas duckdb yfinance requests robin-stocks python-dotenv
```

### Configuration

Set environment variables in `.env` file:

```bash
ALPHAVANTAGE_API_KEY=your_key_here
ROBIN_USER=your_email
ROBIN_PASS=your_password
```

### Usage

```bash
# Full universe backfill
python tick.py

# Test with first 10 tickers
python tick.py --test 10

# Specific tickers only
python tick.py --tickers AAPL MSFT GOOGL

# OHLCV only (skip fundamentals)
python tick.py --no-fundamentals

# Batch processing for large universes
python tick.py --test 5000 --no-fundamentals
```

### Data Storage

- OHLCV: `data_drops/ohlcv_daily/{ticker}.parquet`
- Fundamentals: `data_drops/fundamentals/{statement_type}/{ticker}.parquet`
- State: `data_drops/pipeline_state.duckdb`
- Logs: `logs/pipeline.log`

### Rate Limits

- Yahoo Finance: ~18 tickers per minute (conservative scraping rate)
- Alpha Vantage Free Tier: 25 API calls per day
- With 3 statements per ticker, approximately 8 tickers per day for fundamentals

### Idempotent Design

The script can be safely interrupted and restarted. It automatically:
- Skips already completed tickers
- Retries failed downloads
- Resumes from where it left off

Run the same command daily to gradually complete large datasets.

### Performance

For 10,000 tickers with OHLCV only:
- Expected runtime: 9-11 hours
- Success rate: 95-98%
- Recommended: Run overnight or in batches

### Requirements

#### Python Dependencies
```bash
pip install pandas duckdb yfinance requests robin-stocks python-dotenv
```

#### External Accounts

- Alpha Vantage API key (free tier: 25 calls/day) - Required for fundamentals only
- Robinhood account (optional) - Only needed if using Robinhood fallback

### Installation
```bash
# Clone or download the script
git clone <repository_url>
cd <repository_directory>

# Install dependencies
pip install -r requirements.txt

# Create .env file with credentials
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

#### Environment Variables

Create a `.env` file in the same directory as the script:
```bash
ALPHAVANTAGE_API_KEY=your_key_here
ROBIN_USER=your_email@example.com
ROBIN_PASS=your_password
```

#### Script Configuration

Key settings in the `Config` class:

- `TARGET_TPM`: Tickers per minute (default: 18)
- `START_DATE`: Beginning of historical data range (default: 10 years ago)
- `BATCH_SIZE`: Number of tickers processed before rate-limit sleep (default: 50)
- `MAX_RETRIES`: Retry attempts for failed requests (default: 4)
- `ENABLE_FUNDAMENTALS`: Toggle fundamental data collection (default: True)

#### Universe File

Provide a Parquet file containing the tickers you want to process. Required columns:

- `ticker`: Stock symbol (e.g., "AAPL", "MSFT")
- Additional columns ignored but preserved for reference

Default path: `nyse_common.parquet`

### Usage

#### Basic Commands
```bash
# Full universe backfill (OHLCV + fundamentals)
python tick.py

# Test with first 10 tickers from universe
python tick.py --test 10

# Process specific tickers only
python tick.py --tickers AAPL MSFT GOOGL TSLA

# OHLCV only, skip fundamentals
python tick.py --no-fundamentals

# Custom universe file
python tick.py --universe my_stocks.parquet
```

#### Batch Processing Strategy

For large universes (1000+ tickers), process in batches:
```bash
# Day 1: First 5000 tickers, OHLCV only
nohup python tick.py --test 5000 --no-fundamentals > batch1.log 2>&1 &

# Day 2: Continue from where you left off
nohup python tick.py --no-fundamentals > batch2.log 2>&1 &

# Day 3+: Run fundamentals (25 API calls per day = ~8 tickers)
python tick.py
```

#### Monitoring Progress

Check status in DuckDB:
```python
import duckdb
con = duckdb.connect("data_drops/pipeline_state.duckdb")

# OHLCV status summary
print(con.execute("SELECT status, COUNT(*) FROM ohlcv_runs GROUP BY status").df())

# Fundamentals status summary
print(con.execute("SELECT status, COUNT(*) FROM fund_runs GROUP BY status").df())

# Failed tickers
print(con.execute("SELECT ticker, error FROM ohlcv_runs WHERE status='failed'").df())
```

### Data Storage

#### Directory Structure
```
data_drops/
├── ohlcv_daily/
│   ├── AAPL.parquet
│   ├── MSFT.parquet
│   └── ...
├── fundamentals/
│   ├── income_statement/
│   │   ├── AAPL.parquet
│   │   └── ...
│   ├── balance_sheet/
│   │   └── ...
│   └── cash_flow/
│       └── ...
└── pipeline_state.duckdb

logs/
└── pipeline.log
```

#### Data Schema

OHLCV files contain:
- `Date`: Trading date (datetime)
- `Open`, `High`, `Low`, `Close`: Price levels (float)
- `Adj Close`: Adjusted closing price (float)
- `Volume`: Trading volume (int)
- `ticker`: Symbol (string)

Fundamentals files contain Alpha Vantage raw response data with:
- `ticker`: Symbol (string)
- `period_type`: "annual" or "quarterly"
- `fiscalDateEnding`: Report date
- Statement-specific fields (revenue, assets, cash flow, etc.)

### Rate Limits and Quotas

#### Yahoo Finance (yfinance)

- No official API, uses web scraping
- Conservative default: 18 tickers per minute
- Risk of temporary blocks if exceeded
- No authentication required

#### Alpha Vantage (Free Tier)

- 25 API calls per day
- 3 statements per ticker = ~8 tickers per day maximum
- Resets at midnight UTC
- Upgrade to premium for higher limits

#### Robinhood (Unofficial)

- Undocumented rate limits
- Approximately 5 years of historical data per request
- Requires login and may trigger 2FA
- Use at your own risk (violates TOS)

### Idempotent Design

TIC-K is designed to be interrupted and resumed safely:

- **Database state tracking**: DuckDB records which tickers succeeded, failed, or are pending
- **Atomic writes**: Parquet files written to temp location then moved atomically
- **Skip completed**: Re-running the script automatically skips already-downloaded tickers
- **Retry failures**: Failed tickers are marked and retried on subsequent runs
- **Resume fundamentals**: Alpha Vantage limit reached? Run again tomorrow to continue

This design is essential for collecting data on large universes over multiple days.

### Performance Expectations

#### OHLCV Collection (10,000 tickers)

- Expected runtime: 9-11 hours
- Success rate: 95-98%
- Failed tickers: Usually delisted, invalid symbols, or API issues
- Recommended: Run overnight or split into batches

#### Fundamentals Collection (10,000 tickers)

- Alpha Vantage free tier: ~8 tickers per day
- Total time: ~3.5 years for complete collection
- Recommendation: Process high-priority tickers first
- Alternative: Use paid Alpha Vantage tier or other data sources

### Troubleshooting

#### Common Issues

**"Universe file not found"**
- Ensure `nyse_common.parquet` exists in the script directory
- Or specify custom path with `--universe` flag

**"Empty OHLCV from yfinance"**
- Ticker may be delisted or invalid
- Check ticker symbol is correct
- Try manually: `yf.download("TICKER")`

**"AlphaVantage throttled/error"**
- Daily limit of 25 calls reached
- Wait until next day (resets midnight UTC)
- Script automatically stops and can be resumed

**"robin_stocks not available"**
- Optional dependency not installed
- Only needed if using Robinhood fallback
- Install with: `pip install robin-stocks`

#### Log Files

All activity logged to:
- Console output (real-time)
- `logs/pipeline.log` (persistent)

Log levels:
- INFO: Normal operation, progress updates
- WARNING: Retries, gaps detected, approaching limits
- ERROR: Failed requests, missing data

### Best Practices

#### For Large Universes

1. Start with OHLCV only (`--no-fundamentals`)
2. Process in batches of 5,000-10,000 tickers
3. Run overnight to avoid interruptions
4. Verify success rate before proceeding to fundamentals
5. Prioritize fundamentals for tickers you actively trade

#### For Daily Updates

1. Run incrementally with same universe file
2. Script automatically fetches only new data
3. Schedule via cron for automated daily updates
4. Monitor logs for increased failure rates

#### For Development/Testing

1. Always use `--test` flag with small numbers first
2. Verify output files are created correctly
3. Check DuckDB state matches expectations
4. Test specific tickers with `--tickers` before full runs

### Future Enhancements

Potential improvements for future versions:

- Additional free data sources (Polygon, IEX Cloud, EODHD)
- Parallel processing with worker pools
- Cloud storage backends (S3, GCS, Azure Blob)
- Real-time/intraday data collection
- Data quality validation and cleansing
- Incremental updates (fetch only new dates)
- Dividend and split adjustment handling

### License

This script is provided as-is for educational and research purposes. Users are responsible for complying with data provider terms of service.