# TOOLS.MD

## Table of Contents

- [ticker_tape.py](#ticker_tapepy)
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