from pathlib import Path

import pandas as pd
import pytest
import report

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_generate_pdf_report_creates_a_real_pdf(tmp_path):
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    transformer_id = df.iloc[0]["transformer_id"]
    out_path = report.generate_pdf_report(transformer_id, DATA, out_dir=tmp_path)
    assert out_path.exists()
    assert out_path.suffix == ".pdf"
    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_pdf_report_raises_for_unknown_id():
    with pytest.raises(ValueError):
        report.generate_pdf_report("T9999", DATA, out_dir=None)
