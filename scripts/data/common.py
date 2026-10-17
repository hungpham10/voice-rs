"""Tiện ích chung cho khâu chuẩn bị dữ liệu (Python — chỉ là dev tooling).

Mọi adapter (`prepare.py vivos|common-voice|vivvoice`) đều bắn ra **cùng một**
`manifest_raw.csv`:

    speaker_id,utt_id,wav_path,duration

và ghi mọi câu về wav **16 kHz mono** dưới `<out>/wav/`. `split.py` mới là nơi
gắn cột `split`/`feats_path` để thành hợp đồng CSV cho Rust (§13.6.4d).

Quy ước: Python **không** dính vào runtime suy luận; chỉ tạo dữ liệu cho huấn
luyện.
"""

from __future__ import annotations

import csv
from pathlib import Path

SR = 16000

RAW_HEADER = ["speaker_id", "utt_id", "wav_path", "duration"]
SPLIT_HEADER = ["split", "speaker_id", "utt_id", "wav_path", "feats_path", "duration"]


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_audio_16k(path) -> "list[float]":
    """Nạp bất kỳ định dạng (wav/mp3/...) về mono 16 kHz bằng librosa."""
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def save_wav_16k(path, y) -> None:
    import soundfile as sf

    ensure_dir(Path(path).parent)
    sf.write(str(path), y, SR, subtype="PCM_16")


def prepared_utt(out_dir, speaker_id: str, utt_id: str, src_path) -> "dict | None":
    """Chuẩn hoá một câu về `<out_dir>/wav/<utt_id>.wav`; trả dòng raw manifest."""
    y = load_audio_16k(src_path)
    if y is None or len(y) == 0:
        return None
    wav_path = Path(out_dir) / "wav" / f"{utt_id}.wav"
    save_wav_16k(wav_path, y)
    return {
        "speaker_id": speaker_id,
        "utt_id": utt_id,
        "wav_path": str(wav_path),
        "duration": round(len(y) / SR, 3),
    }


def write_raw_manifest(path, rows) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_HEADER)
        w.writeheader()
        w.writerows(rows)


def read_raw_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
