"""Loading tabular datasets from the file formats users commonly have."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet")

FileLike = Union[str, Path, BinaryIO]


def load_dataset(source: FileLike, filename: str | None = None) -> pd.DataFrame:
    """Load a dataset from a path or a file-like object.

    ``filename`` is required when ``source`` is a file-like object without a
    ``name`` attribute, because the extension decides which reader is used.
    """
    name = filename or getattr(source, "name", None) or str(source)
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext in (".csv", ".tsv", ".txt"):
        df = _read_delimited(source, default_sep="\t" if ext == ".tsv" else None)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(source)
    elif ext == ".json":
        df = pd.read_json(source)
    else:
        df = pd.read_parquet(source)

    if df.empty:
        raise ValueError("The file does not contain any rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_delimited(source: FileLike, default_sep: str | None) -> pd.DataFrame:
    """Read CSV-like text, sniffing the separator (comma, semicolon, tab...)."""
    if hasattr(source, "read"):
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        raw = Path(source).read_bytes()

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if default_sep:
        return pd.read_csv(io.StringIO(text), sep=default_sep)
    # sep=None lets the python engine sniff the delimiter.
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")
