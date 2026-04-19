import pandas as pd
import pytest

from jaguar_reid.submit import validate_submission_format


def _write(path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_valid_submission(tmp_path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, [{"row_id": i, "similarity": 0.5} for i in range(5)])
    _write(sub, [{"row_id": i, "similarity": 0.5} for i in range(5)])
    validate_submission_format(sub, sample)


def test_bad_column_order(tmp_path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, [{"row_id": i, "similarity": 0.5} for i in range(3)])
    pd.DataFrame({"similarity": [0.5] * 3, "row_id": list(range(3))}).to_csv(sub, index=False)
    with pytest.raises(AssertionError):
        validate_submission_format(sub, sample)


def test_bad_row_count(tmp_path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, [{"row_id": i, "similarity": 0.5} for i in range(5)])
    _write(sub, [{"row_id": i, "similarity": 0.5} for i in range(4)])
    with pytest.raises(AssertionError):
        validate_submission_format(sub, sample)


def test_out_of_range(tmp_path) -> None:
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    _write(sample, [{"row_id": i, "similarity": 0.5} for i in range(3)])
    _write(sub, [{"row_id": i, "similarity": 1.5} for i in range(3)])
    with pytest.raises(AssertionError):
        validate_submission_format(sub, sample)
