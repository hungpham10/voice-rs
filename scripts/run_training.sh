#!/usr/bin/env bash
# Chạy trọn quy trình huấn luyện (ĐƯỜNG DUY NHẤT — nhiều câu/nhiều giọng, §13.6.4):
#
#   embed (features ContentVec) → train (adapter, dec đóng băng) → validate (cổng)
#
#   DATA=data/vivos ./scripts/run_training.sh
#
# Biến môi trường (mặc định trong ngoặc):
#   DATA (data/vivos) · TRAIN_MANIFEST ($DATA/train.csv) · VAL_MANIFEST ($DATA/val.csv)
#   OUT_DIR ($DATA/bridge) · CHUNK (512) · BATCH (8) · ACCUM (4) · STEPS (1500) · LR (1e-3)
#   VAL_EVERY (50 — 0 = tắt val)
#   DEVICE (cpu) · BIN_DIR (bin)
#
# Ưu tiên binary đã build trong ./bin/ (Colab: tải bằng ./scripts/fetch_bin.sh),
# ngược lại dùng `cargo run --release`.
# Lưu ý: SEG (cũ) vẫn được chấp nhận như alias của CHUNK để backward compat.
set -euo pipefail

DATA="${DATA:-data/vivos}"
TRAIN="${TRAIN_MANIFEST:-$DATA/train.csv}"
VAL="${VAL_MANIFEST:-$DATA/val.csv}"
OUT="${OUT_DIR:-$DATA/bridge}"
# CHUNK mới; SEG cũ vẫn hoạt động như alias (ưu tiên CHUNK nếu có)
CHUNK="${CHUNK:-${SEG:-512}}"
BATCH="${BATCH:-2}"
# ACCUM: cộng dồn gradient qua N micro-batch → effective batch = BATCH × ACCUM
# memory chỉ theo BATCH. Mặc định 2×4 = 8 (effective batch 8, RAM ~5GB cho CHUNK=512).
ACCUM="${ACCUM:-4}"
# VAL_EVERY=0 → tắt val hoàn toàn (train liên tục, không tốn VRAM eval)
VAL_EVERY="${VAL_EVERY:-50}"
STEPS="${STEPS:-1500}"
LR="${LR:-1e-4}"
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

echo "== 2/3 train (CHUNK=${CHUNK} BATCH=${BATCH}×${ACCUM} STEPS=${STEPS}) =="
run_voice train \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --out-dir "$OUT" \
  --chunk-frames "$CHUNK" --batch "$BATCH" --grad-accum "$ACCUM" \
  --steps "$STEPS" --lr "$LR" --device "$DEVICE" --val-every "$VAL_EVERY"

echo "== 3/3 validate (cổng PASS/FAIL) =="
run_voice validate \
  --val-manifest "$VAL" --adapter "$OUT/adapter_best.bpk" \
  --out-report "$OUT/report.csv" --out-json "$OUT/summary.json" --device "$DEVICE"

echo "✔ xong: adapter + history.csv + report.csv + summary.json ở $OUT"
