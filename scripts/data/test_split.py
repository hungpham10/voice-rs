"""Test chia train/val (chạy: `python -m pytest scripts/data/test_split.py`).

Chỉ phụ thuộc phần thuần-Python của `split.py` (không cần librosa/âm thanh).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from split import is_val, split_rows  # noqa: E402


def make_rows(speakers, per_speaker):
    rows = []
    for s in speakers:
        for i in range(per_speaker):
            rows.append(
                {
                    "speaker_id": s,
                    "utt_id": f"{s}_u{i:03d}",
                    "wav_path": f"wav/{s}_u{i:03d}.wav",
                    "duration": 3.0,
                }
            )
    return rows


def test_split_is_deterministic():
    rows = make_rows(["spkA", "spkB"], 20)
    a = split_rows(rows, 0.1)
    b = split_rows(rows, 0.1)
    assert [r["utt_id"] for r in a[0]] == [r["utt_id"] for r in b[0]]
    assert [r["utt_id"] for r in a[1]] == [r["utt_id"] for r in b[1]]


def test_no_utterance_leaks_between_splits():
    rows = make_rows(["spkA", "spkB"], 20)
    train, val = split_rows(rows, 0.1)
    tr = {r["utt_id"] for r in train}
    va = {r["utt_id"] for r in val}
    assert tr.isdisjoint(va)
    assert tr | va == {r["utt_id"] for r in rows}


def test_every_speaker_has_val_when_enough():
    rows = make_rows(["spkA", "spkB"], 10)
    _, val = split_rows(rows, 0.1)
    assert {r["speaker_id"] for r in val} == {"spkA", "spkB"}


def test_single_utterance_speaker_goes_to_train():
    rows = make_rows(["solo"], 1)
    train, val = split_rows(rows, 0.5)
    assert len(train) == 1 and len(val) == 0


def test_ratio_bounds():
    rows = make_rows(["a"], 100)
    _, val = split_rows(rows, 0.1)
    assert 5 <= len(val) <= 15  # ~10% với dao động hash nhỏ


def test_is_val_is_stable():
    assert is_val("abc", 0.1) == is_val("abc", 0.1)
