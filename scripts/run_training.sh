#!/usr/bin/env bash
# Chạy trọn quy trình huấn luyện/validate:
#   (features nếu thiếu) → VALIDATION (dec vs PyTorch) → TRAINING (adapter).
#
# Ưu tiên BINARY đã build sẵn trong ./bin/ (tải bằng ./scripts/fetch_bin.sh —
# đúng cách chạy trên Colab, không cần Rust). Nếu không có thì tự `cargo run`.
#
#   ./scripts/run_training.sh                 # SMOKE: SEG=86 = 1 s đầu (mặc định, nhanh)
#   SEG=full ./scripts/run_training.sh        # huấn luyện CẢ CÂU (chậm hơn nhiều)
#   SEG=200 STEPS=300 ./scripts/run_training.sh
#
# Yêu cầu trước: ./scripts/fetch_assets.sh (models) + ./scripts/fetch_bin.sh (binaries).
set -euo pipefail

SEG="${SEG:-86}"        # "full"/0 → cả câu
STEPS="${STEPS:-250}"
LR="${LR:-1e-3}"
WAV="${WAV:-sources/422-122949-0031.wav}"
FEATS="${FEATS:-sources/features/422-122949-0031.npy}"
BIN_DIR="${BIN_DIR:-bin}"

# "full" hoặc "0" → 0 (voice-train hiểu 0 = cả câu).
if [ "${SEG}" = "full" ] || [ "${SEG}" = "0" ]; then
  SEG=0
fi

# Chạy binary trong ./bin/ nếu có, ngược lại dùng `cargo run --release`.
run_bin() { # <bin> <pkg> [args...]
  local bin="$1" pkg="$2"; shift 2
  if [ -x "${BIN_DIR}/${bin}" ]; then
    "${BIN_DIR}/${bin}" "$@"
  else
    echo "  (không thấy ${BIN_DIR}/${bin} — dùng cargo run -p ${pkg} --bin ${bin})" >&2
    cargo run --release -q -p "${pkg}" --bin "${bin}" -- "$@"
  fi
}

if [ ! -f sources/vits_ljs_dec/config.json ]; then
  echo "✗ thiếu sources/vits_ljs_dec — chạy ./scripts/fetch_assets.sh trước" >&2
  exit 1
fi

if [ ! -f "${FEATS}" ]; then
  echo "== 1/3 features ContentVec cho ${WAV} =="
  run_bin embed voice-hubert "${WAV}" "${FEATS}"
else
  echo "== 1/3 features đã có: ${FEATS} =="
fi

echo "== 2/3 VALIDATION: dec (Rust) vs PyTorch =="
run_bin verify_dec voice-models | tail -4

if [ "${SEG}" = "0" ]; then
  echo "== 3/3 TRAINING: adapter overfit 1 câu — CẢ CÂU (STEPS=${STEPS} LR=${LR}) =="
  echo "   ⚠️  cả câu rất chậm trên CPU (~33 s/bước); Colab nên để SEG=86 cho nhanh"
  run_bin voice-train voice-train train overfit \
    --wav "${WAV}" --feats "${FEATS}" \
    --steps "${STEPS}" --lr "${LR}" \
    --out-wav test_output.wav --out-adapter sources/features/adapter.bpk
else
  echo "== 3/3 TRAINING: adapter overfit 1 câu — SMOKE ${SEG} frame ≈ $(awk "BEGIN{printf \"%.2f\", ${SEG}*256/22050}") s đầu =="
  echo "   (muốn cả câu: SEG=full ./scripts/run_training.sh)"
  run_bin voice-train voice-train train overfit \
    --wav "${WAV}" --feats "${FEATS}" \
    --seg "${SEG}" --steps "${STEPS}" --lr "${LR}" \
    --out-wav test_output.wav --out-adapter sources/features/adapter.bpk
fi

echo "✔ xong: test_output.wav + sources/features/adapter.bpk"
echo "   render cả câu sau này: ./bin/voice-train render --feats ${FEATS} --adapter sources/features/adapter.bpk --out-wav full.wav"
