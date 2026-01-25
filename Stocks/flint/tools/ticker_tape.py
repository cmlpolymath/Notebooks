#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import io
import os
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Set

import httpx
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

# ============================================================
# ticker_tape.py
#
# Goal (your default “one command” workflow):
#   - Build a deduped “common stock” universe that is:
#       * Listed US exchanges (NYSE/Nasdaq + a few others)
#       * PLUS OTC equities from FINRA CAT symbol master (listingExchange=U)
#   - Adds a column that denotes ticker type (listed vs otc) and exchange
#   - Exports to Parquet so you can move straight into OHLCV scraping
#
# Default run:
#   python ticker_tape.py common
# ============================================================

APP = typer.Typer(add_completion=False, help="Ticker universe builder (listed + optional OTC) for OHLCV modeling")
console = Console()

# -------------------- Sources --------------------
NASDAQ_OTHERLISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
NASDAQ_NASDAQLISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
SEC_EXCH_JSON = "https://www.sec.gov/files/company_tickers_exchange.json"

# FINRA CAT symbol master (public)
CAT_EQUITY_SOD = "https://files.catnmsplan.com/symbol-master/FINRACATReportableEquitySecurities_SOD.txt"
CAT_EQUITY_EOD = "https://files.catnmsplan.com/symbol-master/FINRACATReportableEquitySecurities_EOD.txt"

DEFAULT_TIMEOUT = 45.0

# -------------------- Exchange groups --------------------
class ExchangeGroup(str, Enum):
    all_us = "all-us"  # N, A, P, Q, Z, V (listed) + optional OTC via CAT (U)

# NasdaqTrader-ish exchange letter codes
GROUP_TO_LETTERS = {
    ExchangeGroup.all_us: {"N", "A", "P", "Q", "Z", "V"},
}

# Map SEC exchange strings to our letter codes (for optional cross-check)
SEC_TO_LETTER = {
    "NYSE": "N",
    "NYSE American": "A",
    "NYSE Arca": "P",
    "Nasdaq": "Q",
    "Cboe BZX": "Z",
    "IEX": "V",
}

LETTER_TO_LABEL = {
    "N": "NYSE",
    "A": "NYSE American",
    "P": "NYSE Arca",
    "Q": "Nasdaq",
    "Z": "Cboe BZX",
    "V": "IEX",
    "U": "OTC (CAT)",
}

class OTCMode(str, Enum):
    sod = "sod"
    eod = "eod"
    both = "both"

# -------------------- Common ticker pattern --------------------
def common_regex(include_class_shares: bool) -> re.Pattern:
    # Default: conservative “common” ticker format.
    # If class shares enabled, allow one ".X" suffix (e.g. BRK.B).
    return re.compile(r"^[A-Z]{1,4}(\.[A-Z])?$" if include_class_shares else r"^[A-Z]{1,5}$")


# -------------------- Config --------------------
@dataclass(frozen=True)
class UniverseConfig:
    group: ExchangeGroup = ExchangeGroup.all_us
    include_otc: bool = True
    otc_mode: OTCMode = OTCMode.both
    cross_check_sec: bool = True
    include_class_shares: bool = True
    timeout: float = DEFAULT_TIMEOUT


# -------------------- Helper utils --------------------
def _ua(user_agent_opt: Optional[str]) -> dict:
    ua = user_agent_opt or os.getenv("SEC_UA") or "research/educational (contact: gimmeall@thedata.com)"
    return {"User-Agent": ua}


def _maybe_uvloop(enable: bool):
    if not enable:
        return
    try:
        import uvloop  # type: ignore
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except Exception:
        pass


def _clean_txt_nasdaqtrader(txt: str) -> str:
    # NasdaqTrader symbol dir has a “File Creation Time” footer; drop it.
    return "\n".join(
        ln for ln in txt.splitlines()
        if ln.strip() and not ln.startswith("File Creation Time")
    )


def _clean_txt_cat_keep_header(txt: str) -> str:
    """
    CAT files include a footer like: YYYYMMDDHHMMSS|<recordcount>
    Keep header + rows; drop footer if present.
    """
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if lines and re.match(r"^\d{14}\|\d+$", lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines)


# -------------------- Parsers --------------------
def _parse_otherlisted_to_pl(txt: str) -> pl.DataFrame:
    df = pl.read_csv(
        io.StringIO(_clean_txt_nasdaqtrader(txt)),
        separator="|",
        infer_schema_length=0,
        ignore_errors=True,
    ).rename({
        "ACT Symbol": "ticker",
        "Security Name": "name",
        "Exchange": "exchange",
        "ETF": "etf",
        "Test Issue": "test_issue",
    })

    # Normalize empties
    return df.filter(
        pl.col("ticker").is_not_null()
        & (pl.col("ticker") != "")
        & pl.col("exchange").is_not_null()
    ).select(
        pl.col("ticker").cast(pl.Utf8).str.to_uppercase().alias("ticker"),
        pl.col("name").cast(pl.Utf8).alias("name"),
        pl.col("exchange").cast(pl.Utf8).alias("exchange"),
        pl.col("etf").cast(pl.Utf8).alias("etf"),
        pl.col("test_issue").cast(pl.Utf8).alias("test_issue"),
    )


def _parse_nasdaqlisted_to_pl(txt: str) -> pl.DataFrame:
    df = pl.read_csv(
        io.StringIO(_clean_txt_nasdaqtrader(txt)),
        separator="|",
        infer_schema_length=0,
        ignore_errors=True,
    )
    sym_col = "Symbol" if "Symbol" in df.columns else "NASDAQ Symbol"
    rename = {sym_col: "ticker"}
    if "Security Name" in df.columns:
        rename["Security Name"] = "name"
    if "ETF" in df.columns:
        rename["ETF"] = "etf"
    if "Test Issue" in df.columns:
        rename["Test Issue"] = "test_issue"

    df = df.rename(rename).filter(pl.col("ticker").is_not_null() & (pl.col("ticker") != ""))

    # Nasdaq is primary listing here → exchange 'Q'
    return df.with_columns(
        pl.lit("Q").alias("exchange"),
        pl.col("ticker").cast(pl.Utf8).str.to_uppercase().alias("ticker"),
        pl.col("name").cast(pl.Utf8).alias("name"),
        pl.col("etf").cast(pl.Utf8).alias("etf"),
        pl.col("test_issue").cast(pl.Utf8).alias("test_issue"),
    ).select("ticker", "name", "exchange", "etf", "test_issue")


def _parse_sec_to_pl(sec_json: object) -> pl.DataFrame:
    # Handles both {fields,data} and list-of-dicts; silence orientation warning
    if isinstance(sec_json, dict) and "fields" in sec_json and "data" in sec_json:
        sec = pl.DataFrame(sec_json["data"], schema=sec_json["fields"], orient="row")
    else:
        sec = pl.DataFrame(sec_json)

    cols = {c.lower(): c for c in sec.columns}
    tcol = cols[[k for k in cols if k.startswith("tick")][0]]
    ecol = cols[[k for k in cols if k.startswith("exch")][0]]

    sec = sec.rename({tcol: "ticker", ecol: "exchange"}).select(
        pl.col("ticker").cast(pl.Utf8).str.to_uppercase().alias("ticker"),
        pl.col("exchange").cast(pl.Utf8).alias("exchange"),
    ).unique(subset=["ticker"])

    return sec


def _parse_cat_equity_master(txt: str) -> pl.DataFrame:
    """
    FINRA CAT reportable equity symbol master.
    We keep:
      - symbol
      - issueName
      - listingExchange (U = OTC Equity)
      - testIssueFlag (if present)
    """
    cleaned = _clean_txt_cat_keep_header(txt)
    df = pl.read_csv(
        io.StringIO(cleaned),
        separator="|",
        infer_schema_length=0,
        ignore_errors=True,
    )

    cols = {c.lower(): c for c in df.columns}
    sym = cols.get("symbol")
    name = cols.get("issuename")
    exch = cols.get("listingexchange")
    test = cols.get("testissueflag")  # may exist

    if not sym or not name or not exch:
        # If format changes unexpectedly, return best effort
        return df

    out = df.select(
        pl.col(sym).cast(pl.Utf8).str.to_uppercase().alias("ticker"),
        pl.col(name).cast(pl.Utf8).alias("name"),
        pl.col(exch).cast(pl.Utf8).alias("cat_exchange"),
        (pl.col(test).cast(pl.Utf8) if test else pl.lit(None).cast(pl.Utf8)).alias("test_issue"),
    )
    return out


# -------------------- Filters & merge rules --------------------
def _filter_by_group(df: pl.DataFrame, letters: Set[str]) -> pl.DataFrame:
    return df.filter(pl.col("exchange").is_in(list(letters)))


def _apply_common_filters_listed(df: pl.DataFrame, pat: re.Pattern) -> pl.DataFrame:
    # Common-only: not ETF, not test issue, and ticker matches regex
    return (
        df.filter(
            (pl.col("etf") != "Y")
            & (pl.col("test_issue") != "Y")
            & pl.col("ticker").str.contains(pat.pattern, literal=False)
        )
        .with_columns(pl.col("ticker").str.strip_chars())
        .select("ticker", "name", "exchange")
        .unique(subset=["ticker"])
    )


def _apply_common_filters_otc(df: pl.DataFrame, pat: re.Pattern) -> pl.DataFrame:
    # OTC: keep CAT exchange U, drop test issues, apply regex
    return (
        df.filter(
            (pl.col("test_issue") != "Y") | pl.col("test_issue").is_null()
        )
        .filter(pl.col("ticker").str.contains(pat.pattern, literal=False))
        .with_columns(pl.col("ticker").str.strip_chars())
        .select("ticker", "name")
        .unique(subset=["ticker"])
    )


def _dedupe_with_preference(df: pl.DataFrame) -> pl.DataFrame:
    """
    Dedupes by ticker, preferring:
      1) listed tickers (ticker_type='listed')
      2) then otc (ticker_type='otc')
    """
    # Priority: listed=0, otc=1
    prioritized = df.with_columns(
        pl.when(pl.col("ticker_type") == "listed").then(0).otherwise(1).alias("_priority")
    ).sort(["_priority", "ticker"])

    # Keep first occurrence after sort
    return (
        prioritized.unique(subset=["ticker"], keep="first")
        .drop("_priority")
        .sort("ticker")
    )


# -------------------- Networking --------------------
async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url)
    r.raise_for_status()
    return r.text


async def _fetch_json(client: httpx.AsyncClient, url: str) -> object:
    r = await client.get(url)
    r.raise_for_status()
    return r.json()


async def _fetch_listed_and_sec(timeout: float, sec_headers: dict) -> Tuple[str, str, object]:
    async with httpx.AsyncClient(timeout=timeout) as c_listed, httpx.AsyncClient(timeout=timeout, headers=sec_headers) as c_sec:
        other_task = _fetch_text(c_listed, NASDAQ_OTHERLISTED)
        nasdaq_task = _fetch_text(c_listed, NASDAQ_NASDAQLISTED)
        sec_task = _fetch_json(c_sec, SEC_EXCH_JSON)
        return await asyncio.gather(other_task, nasdaq_task, sec_task)


async def _fetch_otc_cat(timeout: float, mode: OTCMode) -> pl.DataFrame:
    urls: List[str] = []
    if mode in (OTCMode.sod, OTCMode.both):
        urls.append(CAT_EQUITY_SOD)
    if mode in (OTCMode.eod, OTCMode.both):
        urls.append(CAT_EQUITY_EOD)

    async with httpx.AsyncClient(timeout=timeout) as c:
        texts = await asyncio.gather(*(_fetch_text(c, u) for u in urls))

    frames = [_parse_cat_equity_master(t) for t in texts]
    cat = pl.concat(frames, how="vertical", rechunk=True)

    # listingExchange: U = OTC Equity
    if "cat_exchange" in cat.columns:
        cat = cat.filter(pl.col("cat_exchange") == "U")

    return cat.select("ticker", "name", "test_issue").unique(subset=["ticker"])


# -------------------- Universe builder --------------------
async def build_common_universe(cfg: UniverseConfig, sec_headers: dict) -> pl.DataFrame:
    letters = GROUP_TO_LETTERS[cfg.group]
    pat = common_regex(cfg.include_class_shares)

    other_txt, nasdaq_txt, sec_json = await _fetch_listed_and_sec(cfg.timeout, sec_headers)

    other_df = _parse_otherlisted_to_pl(other_txt)
    nasdaq_df = _parse_nasdaqlisted_to_pl(nasdaq_txt)

    listed = pl.concat([other_df, nasdaq_df], how="vertical", rechunk=True)
    listed = _filter_by_group(listed, letters)
    listed_common = _apply_common_filters_listed(listed, pat).with_columns(
        pl.lit("listed").alias("ticker_type"),
        pl.lit("common").alias("security_type"),
        pl.col("exchange").cast(pl.Utf8).alias("exchange"),
        pl.col("exchange").map_elements(lambda x: LETTER_TO_LABEL.get(str(x), str(x)), return_dtype=pl.Utf8).alias("exchange_name"),
        pl.lit("nasdaqtrader").alias("source"),
    ).select("ticker", "name", "exchange", "exchange_name", "ticker_type", "security_type", "source")

    # Optional SEC cross-check (listed only)
    if cfg.cross_check_sec:
        sec = _parse_sec_to_pl(sec_json)
        # Map SEC exchange strings to our letter codes, keep only those within group letters
        sec_keep = (
            sec.with_columns(
                pl.col("exchange").map_elements(lambda x: SEC_TO_LETTER.get(str(x), None), return_dtype=pl.Utf8).alias("ex_letter")
            )
            .filter(pl.col("ex_letter").is_in(list(letters)))
            .select(pl.col("ticker"))
            .unique()
        )
        listed_common = listed_common.join(sec_keep, on="ticker", how="inner")

    frames = [listed_common]

    # Optional OTC union (CAT)
    if cfg.include_otc:
        otc_raw = await _fetch_otc_cat(cfg.timeout, cfg.otc_mode)
        otc_common = _apply_common_filters_otc(otc_raw, pat).with_columns(
            pl.lit("U").alias("exchange"),
            pl.lit(LETTER_TO_LABEL["U"]).alias("exchange_name"),
            pl.lit("otc").alias("ticker_type"),
            pl.lit("common").alias("security_type"),
            pl.lit("finra_cat").alias("source"),
        ).select("ticker", "name", "exchange", "exchange_name", "ticker_type", "security_type", "source")
        frames.append(otc_common)

    universe = pl.concat(frames, how="vertical", rechunk=True)
    universe = _dedupe_with_preference(universe)

    # Final guaranteed schema & order
    return universe.select(
        "ticker",
        "name",
        "ticker_type",      # listed | otc
        "security_type",    # common
        "exchange",         # N/A/P/Q/Z/V/U
        "exchange_name",    # human label
        "source",           # nasdaqtrader | finra_cat
    ).sort("ticker")


# -------------------- CLI outputs --------------------
def _print_preview(df: pl.DataFrame, title: str, head: int):
    table = Table(title=title, header_style="bold")
    for c in df.columns:
        table.add_column(c)
    for row in df.head(head).iter_rows(named=True):
        table.add_row(*(str(row[c]) for c in df.columns))
    console.print(table)


def _write_parquet(df: pl.DataFrame, path: str, compression: str = "zstd"):
    df.write_parquet(path, compression=compression, statistics=True)


# -------------------- Commands --------------------
@APP.command("common")
def cmd_common(
    out: str = typer.Option("universe_common.parquet", help="Output Parquet path"),
    head: int = typer.Option(25, help="Preview rows in terminal"),
    no_otc: bool = typer.Option(False, help="Disable OTC union (FINRA CAT)"),
    no_sec_check: bool = typer.Option(False, help="Disable SEC cross-check for listed tickers"),
    otc_mode: OTCMode = typer.Option(OTCMode.both, help="Which CAT file(s) to use (SOD/EOD/both)"),
    include_class_shares: bool = typer.Option(True, help="Allow class shares like BRK.B"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, help="HTTP timeout seconds"),
    uvloop: bool = typer.Option(True, help="Try to use uvloop if available"),
    sec_user_agent: Optional[str] = typer.Option(None, help="SEC User-Agent header (or set SEC_UA env var)"),
):
    """
    Build a deduped, modeling-friendly universe of common stocks:
      - Listed exchanges (all-us group) + optional OTC (FINRA CAT)
      - Adds ticker_type column (listed vs otc)
      - Writes Parquet to --out
    """
    _maybe_uvloop(uvloop)
    cfg = UniverseConfig(
        group=ExchangeGroup.all_us,
        include_otc=(not no_otc),
        otc_mode=otc_mode,
        cross_check_sec=(not no_sec_check),
        include_class_shares=include_class_shares,
        timeout=timeout,
    )

    df = asyncio.run(build_common_universe(cfg, sec_headers=_ua(sec_user_agent)))

    _print_preview(
        df,
        title=f"Common Universe (listed + OTC={'on' if cfg.include_otc else 'off'})",
        head=head,
    )

    console.print(f"[dim]Total tickers:[/dim] {df.height:,}")
    # Useful quick sanity counts by ticker_type and exchange
    try:
        counts = df.group_by(["ticker_type", "exchange"]).len().sort("len", descending=True)
        _print_preview(counts, title="Counts by ticker_type + exchange", head=50)
    except Exception:
        pass

    _write_parquet(df, out)
    console.print(f"[green]Saved Parquet ->[/green] {out}")


@APP.command("stats")
def cmd_stats(
    inp: str = typer.Argument("universe_common.parquet", help="Parquet file to analyze"),
):
    """Show counts by ticker_type/exchange from an exported parquet."""
    df = pl.read_parquet(inp)
    console.print(f"[dim]Rows:[/dim] {df.height:,}")
    counts = df.group_by(["ticker_type", "exchange"]).len().sort("len", descending=True)
    _print_preview(counts, title="Counts by ticker_type + exchange", head=200)


@APP.command("sample-info")
def cmd_sample_info(
    inp: str = typer.Option("universe_common.parquet", help="Parquet universe file"),
    n: int = typer.Option(10, help="How many tickers to sample"),
    seed: int = typer.Option(42, help="Random seed"),
    max_concurrency: int = typer.Option(8, help="Parallel yfinance .info calls"),
):
    """
    Sample yfinance .info for N random tickers from an existing parquet universe.
    (This keeps universe-building separate from yfinance scraping work.)
    """
    import yfinance as yf

    df = pl.read_parquet(inp)
    tickers = df.get_column("ticker").to_list()

    rnd = random.Random(seed)
    picks = rnd.sample(tickers, k=min(n, len(tickers)))

    async def run():
        sem = asyncio.Semaphore(max_concurrency)

        async def fetch_one(sym: str):
            async with sem:
                def _get():
                    try:
                        return sym, yf.Ticker(sym).info
                    except Exception as e:
                        return sym, {"_error": str(e)}
                return await asyncio.to_thread(_get)

        return await asyncio.gather(*(fetch_one(t) for t in picks))

    infos = asyncio.run(run())

    table = Table(title="yfinance .info sample", header_style="bold")
    table.add_column("Ticker")
    table.add_column("Ticker Type")
    table.add_column("Exchange")
    table.add_column("Label / Error")
    table.add_column("Keys (first 6)")

    meta = {r["ticker"]: r for r in df.select("ticker", "ticker_type", "exchange").iter_rows(named=True)}
    for sym, info in infos:
        m = meta.get(sym, {})
        if "_error" in info:
            table.add_row(sym, m.get("ticker_type", "?"), m.get("exchange", "?"), f"[red]{info['_error']}[/red]", "-")
        else:
            label = info.get("shortName") or info.get("longName") or info.get("quoteType") or ""
            keys = ", ".join(list(info.keys())[:6])
            table.add_row(sym, m.get("ticker_type", "?"), m.get("exchange", "?"), str(label), keys)

    console.print(table)


def main():
    APP()


if __name__ == "__main__":
    main()
