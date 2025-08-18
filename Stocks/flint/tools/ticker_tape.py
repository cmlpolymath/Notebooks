#!/usr/bin/env python3
from __future__ import annotations
import asyncio, io, os, random, re
from enum import Enum
from typing import Optional, Tuple, List, Iterable

import httpx
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

APP = typer.Typer(help="US ticker utilities (NYSE core/all, ETFs, cross-check, yfinance sampling)")
console = Console()

# ---------- Sources ----------
NASDAQ_OTHERLISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
NASDAQ_NASDAQLISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
SEC_EXCH_JSON = "https://www.sec.gov/files/company_tickers_exchange.json"

DEFAULT_TIMEOUT = 30.0

# ---------- Exchange groups ----------
class ExchangeGroup(str, Enum):
    nyse_core = "nyse-core"  # N
    nyse_all  = "nyse-all"   # N, A, P  (NYSE, NYSE American, NYSE Arca)
    all_us    = "all-us"     # N, A, P, Q, Z, V (NYSE family + Nasdaq + Cboe BZX + IEX)

# Map NasdaqTrader letter codes to human labels (informational)
LETTER_TO_LABEL = {
    "N": "NYSE",
    "A": "NYSE American",
    "P": "NYSE Arca",
    "Q": "Nasdaq",
    "Z": "Cboe BZX",
    "V": "IEX",
}

GROUP_TO_LETTERS = {
    ExchangeGroup.nyse_core: {"N"},
    ExchangeGroup.nyse_all: {"N", "A", "P"},
    ExchangeGroup.all_us: {"N", "A", "P", "Q", "Z", "V"},
}

# Map SEC exchange strings to our group filters
SEC_TO_LABEL = {
    "NYSE": "N",
    "NYSE American": "A",
    "NYSE Arca": "P",
    "Nasdaq": "Q",
    "Cboe BZX": "Z",
    "IEX": "V",
}

# ---------- Common ticker pattern ----------
def common_regex(include_class_shares: bool) -> re.Pattern:
    # Conservative <=5 letter tickers OR allow a single ".X" class suffix
    return re.compile(r"^[A-Z]{1,4}(\.[A-Z])?$" if include_class_shares else r"^[A-Z]{1,5}$")

# ---------- Parsing helpers ----------
def _clean_txt(txt: str) -> str:
    return "\n".join(ln for ln in txt.splitlines() if ln.count("|") >= 5 and not ln.startswith("File Creation Time"))

def _parse_otherlisted_to_pl(txt: str) -> pl.DataFrame:
    cleaned = _clean_txt(txt)
    df = pl.read_csv(io.StringIO(cleaned), separator="|", infer_schema_length=0, ignore_errors=True)
    # Standardize columns (otherlisted has ACT Symbol / Security Name / Exchange / ETF / Test Issue)
    rename = {}
    for src, dst in [
        ("ACT Symbol", "ticker"),
        ("Security Name", "name"),
        ("Exchange", "exchange"),
        ("ETF", "etf"),
        ("Test Issue", "test_issue"),
    ]:
        if src in df.columns:
            rename[src] = dst
    df = df.rename(rename)
    # only rows with required fields
    df = df.filter(pl.col("ticker").is_not_null() & (pl.col("ticker") != "") & pl.col("exchange").is_not_null())
    return df.select(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("name").cast(pl.Utf8),
        pl.col("exchange").cast(pl.Utf8),
        pl.col("etf").cast(pl.Utf8).alias("etf"),
        pl.col("test_issue").cast(pl.Utf8).alias("test_issue"),
    )

def _parse_nasdaqlisted_to_pl(txt: str) -> pl.DataFrame:
    cleaned = _clean_txt(txt)
    df = pl.read_csv(io.StringIO(cleaned), separator="|", infer_schema_length=0, ignore_errors=True)
    # Columns can be "Symbol" or "NASDAQ Symbol"; harmonize
    sym_col = "Symbol" if "Symbol" in df.columns else "NASDAQ Symbol"
    rename = {sym_col: "ticker"}
    if "Security Name" in df.columns:
        rename["Security Name"] = "name"
    if "ETF" in df.columns:
        rename["ETF"] = "etf"
    if "Test Issue" in df.columns:
        rename["Test Issue"] = "test_issue"
    df = df.rename(rename)
    df = df.filter(pl.col("ticker").is_not_null() & (pl.col("ticker") != ""))
    # Nasdaq primary listing → mark as exchange 'Q'
    df = df.with_columns(
        pl.lit("Q").alias("exchange"),
        pl.col("ticker").cast(pl.Utf8),
        pl.col("name").cast(pl.Utf8),
        pl.col("etf").cast(pl.Utf8),
        pl.col("test_issue").cast(pl.Utf8),
    )
    return df.select("ticker", "name", "exchange", "etf", "test_issue")

def _filter_by_group(df: pl.DataFrame, letters: set[str]) -> pl.DataFrame:
    return df.filter(pl.col("exchange").is_in(list(letters)))

def _filter_common_only(df: pl.DataFrame, pattern: re.Pattern) -> pl.DataFrame:
    return (
        df.filter(
            (pl.col("etf") != "Y")
            & (pl.col("test_issue") != "Y")
            & pl.col("ticker").str.contains(pattern.pattern, literal=False)
        )
        .with_columns(pl.col("ticker").str.strip_chars())
        .unique(subset=["ticker"])
        .sort("ticker")
    )

def _sec_filter_by_group(df: pl.DataFrame, letters: set[str]) -> pl.DataFrame:
    # Map SEC exchange strings to our letters, then filter
    if "exchange" not in df.columns:
        return df
    mapped = df.with_columns(
        pl.col("exchange").map_elements(lambda x: SEC_TO_LABEL.get(str(x), None), return_dtype=pl.Utf8).alias("ex_letter")
    )
    return mapped.filter(pl.col("ex_letter").is_in(list(letters))).drop("ex_letter")

# ---------- Networking ----------
async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url)
    r.raise_for_status()
    return r.text

async def _fetch_json(client: httpx.AsyncClient, url: str) -> object:
    r = await client.get(url)
    r.raise_for_status()
    return r.json()

async def _fetch_all(timeout: float, sec_headers: dict) -> Tuple[str, str, object]:
    async with httpx.AsyncClient(timeout=timeout) as c1, httpx.AsyncClient(timeout=timeout, headers=sec_headers) as c2:
        other_task = _fetch_text(c1, NASDAQ_OTHERLISTED)
        nasdaq_task = _fetch_text(c1, NASDAQ_NASDAQLISTED)
        sec_task = _fetch_json(c2, SEC_EXCH_JSON)
        other_txt, nasdaq_txt, sec_json = await asyncio.gather(other_task, nasdaq_task, sec_task)
    return other_txt, nasdaq_txt, sec_json

# ---------- Public APIs ----------
async def get_universe(
    group: ExchangeGroup,
    include_class_shares: bool,
    common_only: bool,
    timeout: float,
    sec_headers: dict,
    cross_check: bool,
) -> pl.DataFrame:
    letters = GROUP_TO_LETTERS[group]
    other_txt, nasdaq_txt, sec_json = await _fetch_all(timeout=timeout, sec_headers=sec_headers)

    other_df = _parse_otherlisted_to_pl(other_txt)
    nasdaq_df = _parse_nasdaqlisted_to_pl(nasdaq_txt)

    combined = pl.concat([other_df, nasdaq_df], how="vertical", rechunk=True)
    combined = _filter_by_group(combined, letters)

    if common_only:
        combined = _filter_common_only(combined, common_regex(include_class_shares))

    combined = combined.select("ticker", "name", "exchange").unique(subset=["ticker"])

    if not cross_check:
        return combined.sort("ticker")

    # SEC cross-check: keep only those whose SEC exchange is in the same group
    # Handle both {fields,data} and list-of-dicts shapes and silence orientation warning
    if isinstance(sec_json, dict) and "fields" in sec_json and "data" in sec_json:
        sec_df = pl.DataFrame(sec_json["data"], schema=sec_json["fields"], orient="row")
    else:
        sec_df = pl.DataFrame(sec_json)

    # normalize SEC columns
    cols = {c.lower(): c for c in sec_df.columns}
    tcol = cols[[k for k in cols if k.startswith("tick")][0]]
    ecol = cols[[k for k in cols if k.startswith("exch")][0]]
    name_candidates = [k for k in cols if k.startswith("name") or k.startswith("title")]
    ncol = cols[name_candidates[0]] if name_candidates else tcol

    sec_df = sec_df.rename({tcol: "ticker", ecol: "exchange", ncol: "name"}).select(
        pl.col("ticker").str.to_uppercase().alias("ticker"),
        pl.col("name"),
        pl.col("exchange"),
    )
    sec_df = _sec_filter_by_group(sec_df, letters).unique(subset=["ticker"])

    cross = combined.join(sec_df.select("ticker"), on="ticker", how="inner")
    return cross.select("ticker", "name", "exchange").sort("ticker")

async def get_etfs(
    group: ExchangeGroup,
    search: Optional[str],
    timeout: float,
) -> pl.DataFrame:
    letters = GROUP_TO_LETTERS[group]
    # Pull both files; ETFs live in both
    async with httpx.AsyncClient(timeout=timeout) as client:
        other_txt, nasdaq_txt = await asyncio.gather(
            _fetch_text(client, NASDAQ_OTHERLISTED),
            _fetch_text(client, NASDAQ_NASDAQLISTED),
        )
    other_df = _parse_otherlisted_to_pl(other_txt)
    nasdaq_df = _parse_nasdaqlisted_to_pl(nasdaq_txt)

    df = pl.concat([other_df, nasdaq_df], how="vertical", rechunk=True)
    df = _filter_by_group(df, letters).filter(pl.col("etf") == "Y").select("ticker", "name", "exchange").unique(subset=["ticker"])
    if search:
        needle = search.lower()
        df = df.filter(pl.col("name").str.to_lowercase().str.contains(needle, literal=True))
    return df.sort("ticker")

async def sample_yf_infos(
    tickers: List[str],
    n: int = 10,
    seed: Optional[int] = 42,
    max_concurrency: int = 8,
):
    import yfinance as yf
    rnd = random.Random(seed)
    picks = rnd.sample(tickers, k=min(n, len(tickers)))
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

# ---------- Output helpers ----------
def _print_df(df: pl.DataFrame, title: str, head: int):
    table = Table(title=title, header_style="bold")
    for c in df.columns:
        table.add_column(c)
    for row in df.head(head).iter_rows(named=True):
        table.add_row(*(str(row[c]) for c in df.columns))
    console.print(table)

def _write_parquet(df: pl.DataFrame, path: str, compression: str = "zstd"):
    df.write_parquet(path, compression=compression, statistics=True)

def _ua(user_agent_opt: Optional[str]) -> dict:
    ua = user_agent_opt or os.getenv("SEC_UA") or "research/educational (contact: example@example.com)"
    return {"User-Agent": ua}

def _maybe_uvloop(enable: bool):
    if not enable:
        return
    try:
        import uvloop  # type: ignore
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except Exception:
        pass

# ---------- Commands ----------
@APP.command("common")
def cmd_common(
    head: int = typer.Option(25, help="Rows to preview"),
    parquet: Optional[str] = typer.Option(None, help="Write full result to .parquet"),
    group: ExchangeGroup = typer.Option(ExchangeGroup.nyse_core, help="Which exchanges to include"),
    include_class_shares: bool = typer.Option(False, help="Allow class shares like BRK.B"),
    common_only: bool = typer.Option(True, help="Filter to common-only tickers"),
    cross_check: bool = typer.Option(True, help="Cross-check with SEC for the chosen group"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, help="HTTP timeout (seconds)"),
    uvloop: bool = typer.Option(True, help="Try to use uvloop if available"),
    sec_user_agent: Optional[str] = typer.Option(None, help="SEC User-Agent header (or set SEC_UA env var)"),
):
    """Show common stocks for the chosen exchange group (SEC cross-checked by default)."""
    _maybe_uvloop(uvloop)
    df = asyncio.run(get_universe(
        group=group,
        include_class_shares=include_class_shares,
        common_only=common_only,
        timeout=timeout,
        sec_headers=_ua(sec_user_agent),
        cross_check=cross_check,
    ))
    _print_df(df, f"Common stocks ({group})", head)
    console.print(f"[dim]Total rows: {df.height}[/dim]")
    if parquet:
        _write_parquet(df, parquet)
        console.print(f"[green]Saved Parquet ->[/green] {parquet}")

@APP.command("etfs")
def cmd_etfs(
    search: Optional[str] = typer.Option(None, help="Case-insensitive name search (e.g. 'S&P')"),
    head: int = typer.Option(25, help="Rows to preview"),
    parquet: Optional[str] = typer.Option(None, help="Write full result to .parquet"),
    group: ExchangeGroup = typer.Option(ExchangeGroup.nyse_all, help="Exchanges to include (nyse-all covers most ETFs)"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, help="HTTP timeout (seconds)"),
    uvloop: bool = typer.Option(True, help="Try to use uvloop if available"),
):
    """List ETFs for the chosen exchange group (defaults to nyse-all to include Arca like SPY)."""
    _maybe_uvloop(uvloop)
    df = asyncio.run(get_etfs(group=group, search=search, timeout=timeout))
    _print_df(df, f"ETFs ({group})", head)
    console.print(f"[dim]Total rows: {df.height}[/dim]")
    if parquet:
        _write_parquet(df, parquet)
        console.print(f"[green]Saved Parquet ->[/green] {parquet}")

@APP.command("sample-info")
def cmd_sample_info(
    n: int = typer.Option(10, help="How many tickers to sample"),
    seed: Optional[int] = typer.Option(42, help="Random seed"),
    max_concurrency: int = typer.Option(8, help="Parallel yfinance .info calls"),
    group: ExchangeGroup = typer.Option(ExchangeGroup.nyse_all, help="Pick universe first (nyse-all recommended)"),
    include_class_shares: bool = typer.Option(False, help="Allow class shares like BRK.B"),
    common_only: bool = typer.Option(True, help="Filter to common-only"),
    cross_check: bool = typer.Option(True, help="SEC cross-check"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, help="HTTP timeout (seconds)"),
    uvloop: bool = typer.Option(True, help="Try to use uvloop if available"),
    sec_user_agent: Optional[str] = typer.Option(None, help="SEC User-Agent header (or set SEC_UA env var)"),
):
    """Fetch yfinance .info for a random sample from the chosen universe."""
    _maybe_uvloop(uvloop)
    universe = asyncio.run(get_universe(
        group=group,
        include_class_shares=include_class_shares,
        common_only=common_only,
        timeout=timeout,
        sec_headers=_ua(sec_user_agent),
        cross_check=cross_check,
    ))
    tickers = universe.get_column("ticker").to_list()
    infos = asyncio.run(sample_yf_infos(tickers, n=n, seed=seed, max_concurrency=max_concurrency))
    table = Table(title=f"yfinance .info sample ({group})", header_style="bold")
    table.add_column("Ticker"); table.add_column("Label / Error"); table.add_column("Example Keys")
    for sym, info in infos:
        if "_error" in info:
            table.add_row(sym, f"[red]{info['_error']}[/red]", "-")
        else:
            label = info.get("shortName") or info.get("longName") or info.get("quoteType") or ""
            some_keys = ", ".join(list(info.keys())[:6])
            table.add_row(sym, label, some_keys)
    console.print(table)

def main():
    APP()

if __name__ == "__main__":
    main()



commands = """
# Preview NYSE core common (strict NYSE only), 25 rows
python ticker_tape.py common --head 25 --group nyse-core

# Include class shares and save to Parquet for *all NYSE* (NYSE + Arca + American)
python ticker_tape.py common --group nyse-all --include-class-shares --parquet nyse_common.parquet

# Skip SEC cross-check (union of NasdaqTrader files only)
python ticker_tape.py common --no-cross-check --group nyse-all

# List ETFs (Arca-heavy) and search
python ticker_tape.py etfs --group nyse-all
python ticker_tape.py etfs --group nyse-all --search "S&P" --parquet nyse_etfs.parquet

# Sample 10 tickers’ yfinance info from the NYSE-all universe
python ticker_tape.py sample-info --group nyse-all --n 10 --seed 42 --max-concurrency 8

"""