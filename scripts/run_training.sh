#!/usr/bin/env bash
# Chạy trọn quy trình huấn luyện (ĐƯỜNG DUY NHẤT — nhiều câu/nhiều giọng, §13.6.4):
#
#   embed (features ContentVec) → train (adapter, dec đóng băng) → validate (cổng)
#
#   DATA=data/vivos ./scripts/run_training.sh
#
# Biến môi trường (mặc định trong ngoặc):
#   DATA (data/vivos) · TRAIN_MANIFEST ($DATA/train.csv) · VAL_MANIFEST ($DATA/val.csv)
#   OUT_DIR ($DATA/bridge) · SEG (86) · BATCH (8) · STEPS (1500) · LR (1e-3)
#   DEVICE (cpu) · BIN_DIR (bin)
#
# Ưu tiên binary đã build trong ./bin/ (Colab: tải bằng ./scripts/fetch_bin.sh),
# ngược lại dùng `cargo run --release`.
set -euo pipefail

DATA="${DATA:-data/vivos}"
TRAIN="${TRAIN_MANIFEST:-$DATA/train.csv}"
VAL="${VAL_MANIFEST:-$DATA/val.csv}"
OUT="${OUT_DIR:-$DATA/bridge}"
SEG="${SEG:-86}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-1500}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cpu}"
BIN_DIR="${BIN_DIR:-bin}"

run_voice() { # [args...]
  if [ -x "${BIN_DIR}/voice" ]; then
    "${BIN_DIR}/voice" "$@"
  else
    cargo run --release -q -p voice-train --bin voice -- "$@"
  fi
}

if [ ! -f sources/vits_ljs_dec/config.json ]; then
  echo "✗ thiếu sources/vits_ljs_dec — chạy ./scripts/fetch_assets.sh trước" >&2
  exit 1
fi
if [ ! -f "$TRAIN" ] || [ ! -f "$VAL" ]; then
  echo "✗ thiếu manifest ($TRAIN / $VAL) — chạy scripts/data/prepare.py + split.py trước" >&2
  exit 1
fi

echo "== 1/3 features ContentVec (voice embed) =="
run_voice embed --manifest "$TRAIN" --skip-existing --device "$DEVICE"
run_voice embed --manifest "$VAL" --skip-existing --device "$DEVICE"

echo "== 2/3 train (SEG=${SEG} BATCH=${BATCH} STEPS=${STEPS}) =="
run_voice train \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --out-dir "$OUT" \
  --seg "$SEG" --batch "$BATCH" --steps "$STEPS" --lr "$LR" --device "$DEVICE"

echo "== 3/3 validate (cổng PASS/FAIL) =="
run_voice validate \
  --val-manifest "$VAL" --adapter "$OUT/adapter_best.bpk" \
  --out-report "$OUT/report.csv" --out-json "$OUT/summary.json" --device "$DEVICE"

echo "✔ xong: adapter + history.csv + report.csv + summary.json ở $OUT"
