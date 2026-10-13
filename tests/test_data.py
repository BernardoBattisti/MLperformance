import io

import pandas as pd
import pytest

from mlperformance.data import load_dataset


def test_sniffs_semicolon_separator():
    raw = b"a;b;c\n1;2;x\n3;4;y\n"
    df = load_dataset(io.BytesIO(raw), filename="data.csv")
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2


def test_latin1_csv():
    raw = "nome,valor\nJoão,1\nAção,2\n".encode("latin-1")
    df = load_dataset(io.BytesIO(raw), filename="data.csv")
    assert df["nome"].tolist() == ["João", "Ação"]


def test_excel_and_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    df.to_excel(tmp_path / "d.xlsx", index=False)
    df.to_parquet(tmp_path / "d.parquet")
    pd.testing.assert_frame_equal(load_dataset(tmp_path / "d.xlsx"), df)
    pd.testing.assert_frame_equal(load_dataset(tmp_path / "d.parquet"), df)


def test_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(io.BytesIO(b""), filename="data.pdf")
