#!/usr/bin/env bash
# Chạy trọn **Luồng 1** (train bridge) từ dataset đã prepare + split (§13.6.4).
#
#   DATA=data/vivos ./scripts/run_bridge_training.sh
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

run_bin() { # <bin> <pkg> [args...]
  local bin="$1" pkg="$2"; shift 2
  if [ -x "${BIN_DIR}/${bin}" ]; then
    "${BIN_DIR}/${bin}" "$@"
  else
    cargo run --release -q -p "${pkg}" --bin "${bin}" -- "$@"
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

echo "== 1/3 features ContentVec (embed_batch) =="
run_bin embed_batch voice-hubert --manifest "$TRAIN" --skip-existing
run_bin embed_batch voice-hubert --manifest "$VAL" --skip-existing

echo "== 2/3 train bridge (SEG=${SEG} BATCH=${BATCH} STEPS=${STEPS}) =="
run_bin voice-train voice-train train bridge \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --out-dir "$OUT" \
  --seg "$SEG" --batch "$BATCH" --steps "$STEPS" --lr "$LR" --device "$DEVICE"

echo "== 3/3 eval (cổng PASS/FAIL) =="
run_bin voice-train voice-train eval \
  --val-manifest "$VAL" --adapter "$OUT/adapter_best.bpk" \
  --out-report "$OUT/report.csv" --out-json "$OUT/summary.json" --device "$DEVICE"

echo "✔ xong: adapter + history.csv + report.csv + summary.json ở $OUT"
