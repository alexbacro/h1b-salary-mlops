"""Stage 1a: convert each raw DOL Excel file into a Parquet file with the canonical schema.

Reading a ~100 MB .xlsx takes minutes, so each file is converted once and cached
in data/interim/; re-runs skip files whose Parquet is newer than the source.

Usage:
    python -m h1b.ingest            # convert everything in data/raw/
    python -m h1b.ingest --force    # re-convert even if cached
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from h1b.config import get_paths, load_params
from h1b.schema import REQUIRED_COLUMNS, canonical_name

log = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when a raw file is missing columns the pipeline requires."""


def read_lca_excel(path: Path) -> pd.DataFrame:
    """Read one disclosure file, keeping only canonical columns (renamed).

    Everything is read as strings: DOL mixes numbers, '$85,000.00' and blanks in
    the same column, which would otherwise produce mixed-type Parquet columns.
    Typing/parsing happens in h1b.clean.
    """
    # calamine (Rust) is ~10x faster than openpyxl on these files.
    df = pd.read_excel(
        path,
        engine="calamine",
        dtype=str,
        usecols=lambda c: canonical_name(c) is not None,
    )
    rename = {c: canonical_name(c) for c in df.columns}
    df = df.rename(columns=rename)
    # If a file somehow has both "X" and "X_1", keep the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise SchemaError(f"{path.name} is missing required columns: {missing}")

    df["SOURCE_FILE"] = path.name
    return df


def convert_all(raw_dir: Path, interim_dir: Path, force: bool = False) -> list[Path]:
    """Convert every .xlsx in raw_dir to interim_dir/<stem>.parquet. Returns the Parquet paths."""
    sources = sorted(p for p in raw_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if not sources:
        raise FileNotFoundError(f"No .xlsx files in {raw_dir}. See data/README.md for download links.")

    outputs = []
    for src in sources:
        out = interim_dir / f"{src.stem}.parquet"
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime and not force:
            log.info("cached   %s", out.name)
        else:
            log.info("reading  %s", src.name)
            df = read_lca_excel(src)
            df.to_parquet(out, index=False)
            log.info("wrote    %s (%d rows)", out.name, len(df))
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-convert files even if a cached Parquet exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = get_paths(load_params())
    convert_all(paths.raw, paths.interim, force=args.force)


if __name__ == "__main__":
    main()
