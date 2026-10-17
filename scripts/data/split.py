#!/usr/bin/env python3
"""Chia train/val **theo từng câu, trong từng giọng** và ghi hợp đồng CSV.

    python scripts/data/split.py \
        --manifest data/vivos/manifest_raw.csv \
        --out-dir  data/vivos \
        --val-ratio 0.1

Quy tắc:
  - gán `val` nếu `int(sha256(utt_id)[:8], 16) % 10000 < val_ratio * 10000`
    (tất định, đổi máy vẫn ra cùng kết quả);
  - bảo đảm **mỗi giọng có ≥1 câu train và ≥1 câu val** (giọng lẻ để train);
  - ghi `train.csv` + `val.csv` với cột:
    `split,speaker_id,utt_id,wav_path,feats_path,duration`.

`feats_path` mặc định `<out-dir>/features/<utt_id>.npy` — chính nơi
`embed_batch` sẽ ghi features (Rust chỉ đọc, không tự đoán tên).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import SPLIT_HEADER, ensure_dir, read_raw_manifest  # noqa: E402


def is_val(utt_id: str, val_ratio: float) -> bool:
    h = int(hashlib.sha256(utt_id.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return h < int(round(val_ratio * 10000))


def split_rows(rows, val_ratio: float):
    """Trả (train, val) đã bảo đảm mỗi giọng có cả hai phía (khi đủ câu)."""
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["speaker_id"], []).append(r)

    train, val = [], []
    for spk in sorted(by_spk):
        items = sorted(by_spk[spk], key=lambda r: r["utt_id"])
        tr = [r for r in items if not is_val(r["utt_id"], val_ratio)]
        va = [r for r in items if is_val(r["utt_id"], val_ratio)]

        if len(items) == 1:
            tr, va = items, []
        elif not va:  # chưa có val → lấy 1 câu có hash nhỏ nhất
            moved = min(tr, key=lambda r: hashlib.sha256(r["utt_id"].encode()).hexdigest())
            tr.remove(moved)
            va = [moved]
        elif not tr:  # chưa có train → lấy 1 câu có hash lớn nhất
            moved = max(va, key=lambda r: hashlib.sha256(r["utt_id"].encode()).hexdigest())
            va.remove(moved)
            tr = [moved]

        train.extend(tr)
        val.extend(va)
    return train, val


def write_split(path, rows, features_dir: Path, split: str):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SPLIT_HEADER)
        for r in rows:
            feats = features_dir / f"{r['utt_id']}.npy"
            w.writerow(
                [split, r["speaker_id"], r["utt_id"], r["wav_path"], str(feats), r["duration"]]
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest_raw.csv (từ prepare.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--features-dir", default=None, help="mặc định <out-dir>/features")
    args = ap.parse_args()

    if not (0.0 <= args.val_ratio < 1.0):
        raise SystemExit("✗ --val-ratio phải trong [0, 1)")

    rows = read_raw_manifest(args.manifest)
    if not rows:
        raise SystemExit("✗ manifest rỗng")

    train, val = split_rows(rows, args.val_ratio)
    out = Path(args.out_dir)
    feats = Path(args.features_dir) if args.features_dir else out / "features"

    write_split(out / "train.csv", train, feats, "train")
    write_split(out / "val.csv", val, feats, "val")
    ensure_dir(feats)

    n_spk = len({r["speaker_id"] for r in rows})
    val_spk = len({r["speaker_id"] for r in val})
    print(
        f"✔ {len(rows)} câu · {n_spk} giọng → train {len(train)} / val {len(val)}"
        f" (val phủ {val_spk}/{n_spk} giọng)"
    )
    print(f"  {out / 'train.csv'}")
    print(f"  {out / 'val.csv'}")
    print(f"  features → {feats}")
    if val_spk < n_spk:
        print("  ⚠ có giọng không góp câu val nào (quá ít câu)")


if __name__ == "__main__":
    main()
