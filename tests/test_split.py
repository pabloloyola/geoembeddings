from __future__ import annotations

import pandas as pd

from geoembeddings.data import _timestamp_split


def test_timestamp_split_is_strictly_forward() -> None:
    train_end = pd.Timestamp("2026-04-08T00:00:00Z")
    validation_end = pd.Timestamp("2026-04-11T00:00:00Z")
    assert _timestamp_split(pd.Timestamp("2026-04-07T12:00:00Z"), train_end, validation_end) == "train"
    assert _timestamp_split(train_end, train_end, validation_end) == "train"
    assert _timestamp_split(pd.Timestamp("2026-04-09T00:00:00Z"), train_end, validation_end) == "validation"
    assert _timestamp_split(pd.Timestamp("2026-04-12T00:00:00Z"), train_end, validation_end) == "test"

