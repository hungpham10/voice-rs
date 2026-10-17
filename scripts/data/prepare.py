#!/usr/bin/env python3
"""Chuẩn bị dataset → `manifest_raw.csv` (wav 16 kHz mono).

Ba adapter, tất cả ra **cùng một** manifest:

    python scripts/data/prepare.py vivos          --src <dir>  --out data/vivos
    python scripts/data/prepare.py common-voice   --src <dir>  --out data/cv
    python scripts/data/prepare.py vivvoice       --src <dir>  --out data/vivvoice

Tuỳ chọn:
    --limit-per-speaker N   chỉ lấy N câu đầu mỗi giọng (thử nhanh).
    --download              (chỉ vivos) tải + giải nén từ --url trước.

Sau đó chia train/val: `python scripts/data/split.py --manifest ... --out-dir ...`.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import prepared_utt, write_raw_manifest  # noqa: E402

# Mirror công khai của VIVOS (AILAB). File LFS nằm trong thư mục `data/` của repo
# HF. Có thể override bằng --url.
VIVOS_URL = "https://huggingface.co/datasets/AILAB-VNUHCM/vivos/resolve/main/data/vivos.tar.gz"


def _take(rows, limit):
    if limit and limit > 0:
        return rows[:limit]
    return rows


def _walk_by_folder(src: Path, limit):
    """Mẫu chung: mỗi file wav/mp3 dưới `<src>/<speaker>/...` → speaker = thư mục cha."""
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".wav", ".mp3", ".flac"))
    seen = {}
    for p in files:
        spk = p.parent.name
        seen.setdefault(spk, []).append(p)
    out = []
    for spk in sorted(seen):
        out.extend((spk, p) for p in _take(sorted(seen[spk]), limit))
    return out


def cmd_vivos(args):
    src = Path(args.src) if args.src else None
    if args.download:
        src = _download_vivos(args.url, Path(args.out).parent / "_raw")
    if not src or not src.exists():
        raise SystemExit("✗ cần --src <thư mục VIVOS đã giải nén> hoặc --download")

    tasks = _walk_by_folder(src, args.limit_per_speaker)
    _run(tasks, args)


def cmd_vivvoice(args):
    src = Path(args.src)
    tasks = _walk_by_folder(src, args.limit_per_speaker)
    _run(tasks, args)


def cmd_common_voice(args):
    src = Path(args.src)
    tsv = src / "validated.tsv"
    if not tsv.exists():
        raise SystemExit(f"✗ không thấy {tsv} (Common Voice cần clips/ + validated.tsv)")
    clips = src / "clips"

    import csv as _csv

    by_spk = {}
    with open(tsv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f, delimiter="\t"):
            path = row.get("path")
            spk = row.get("client_id")
            if not path or not spk:
                continue
            by_spk.setdefault(spk, []).append(clips / path)

    tasks = []
    for spk in sorted(by_spk):
        tasks.extend((spk, p) for p in _take(sorted(by_spk[spk]), args.limit_per_speaker))
    _run(tasks, args)


def _run(tasks, args):
    out = Path(args.out)
    rows = []
    for i, (spk, path) in enumerate(tasks, 1):
        utt_id = f"{spk}_{path.stem}"
        try:
            row = prepared_utt(out, spk, utt_id, path)
        except Exception as e:  # noqa: BLE001 — dữ liệu thô hay hỏng, bỏ qua + báo
            print(f"  ⚠ bỏ {path}: {e}", file=sys.stderr)
            continue
        if row:
            rows.append(row)
        if i % 200 == 0:
            print(f"  ... {i}/{len(tasks)}")

    manifest = out / "manifest_raw.csv"
    write_raw_manifest(manifest, rows)
    speakers = {r["speaker_id"] for r in rows}
    print(f"✔ {len(rows)} câu · {len(speakers)} giọng → {manifest}")


def _download_vivos(url: str, dest: Path) -> Path:
    """Tải + giải nén VIVOS. Tải ra `.part` rồi đổi tên để chạy lại được, có retry."""
    import time
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / "vivos.tar.gz"
    if not tar.exists():
        part = dest / "vivos.tar.gz.part"
        last_err = None
        for attempt in range(1, 4):
            try:
                print(f"→ tải {url} (lần {attempt}/3)")
                urllib.request.urlretrieve(url, part)
                part.rename(tar)
                break
            except Exception as e:  # noqa: BLE001 — mạng chập chờn, thử lại
                last_err = e
                print(f"  ⚠ lỗi tải: {e}", file=sys.stderr)
                time.sleep(3 * attempt)
        else:
            raise SystemExit(f"✗ không tải được {url}\n  lỗi cuối: {last_err}")
        print(f"  ⤷ {tar.stat().st_size / 1e6:.0f} MB")

    print(f"→ giải nén vào {dest}")
    with tarfile.open(tar) as t:
        try:
            t.extractall(dest, filter="data")  # Python ≥3.12: tránh cảnh báo/không an toàn
        except TypeError:
            t.extractall(dest)

    # Thư mục VIVOS sau giải nén thường là `vivos/waves/...`.
    for base in (dest, *[p for p in dest.iterdir() if p.is_dir()]):
        if (base / "waves").exists():
            return base
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--out", required=True, help="thư mục đích (data/<name>)")
        p.add_argument("--limit-per-speaker", type=int, default=0, help="chỉ lấy N câu/giọng")

    p = sub.add_parser("vivos", help="VIVOS (16 kHz, cần --src hoặc --download)")
    p.add_argument("--src", default=None)
    p.add_argument("--url", default=VIVOS_URL)
    p.add_argument("--download", action="store_true")
    add_common(p)
    p.set_defaults(func=cmd_vivos)

    p = sub.add_parser("common-voice", help="Common Voice vi (clips/ + validated.tsv)")
    p.add_argument("--src", required=True)
    add_common(p)
    p.set_defaults(func=cmd_common_voice)

    p = sub.add_parser("vivvoice", help="viVoice (thư mục <speaker>/*.wav)")
    p.add_argument("--src", required=True)
    add_common(p)
    p.set_defaults(func=cmd_vivvoice)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
