#!/usr/bin/env bash
# Chạy trọn quy trình trên máy đã có `sources/` (từ `scripts/fetch_assets.sh`):
#   build → (features nếu thiếu) → VALIDATION (dec vs PyTorch) → TRAINING (adapter).
#
#   SEG=86 STEPS=250 LR=1e-3 ./scripts/run_training.sh
#
# Mặc định: bundle đã có sẵn features của `samples/422-122949-0031.wav`.
set -euo pipefail

SEG="${SEG:-86}"
STEPS="${STEPS:-250}"
LR="${LR:-1e-3}"
WAV="${WAV:-sources/422-122949-0031.wav}"
FEATS="${FEATS:-sources/features/422-122949-0031.npy}"

if [ ! -f sources/onnx_hubert_new/model.onnx ]; then
  echo "✗ thiếu sources/onnx_hubert_new/model.onnx — chạy ./scripts/fetch_assets.sh trước" >&2
  exit 1
fi

echo "== 1/4 build (voice-models, voice-hubert, voice-train) =="
cargo build --release -p voice-models -p voice-hubert -p voice-train

if [ ! -f "${FEATS}" ]; then
  echo "== 2/4 features ContentVec cho ${WAV} =="
  cargo run --release -p voice-hubert --example embed -- "${WAV}" "${FEATS}"
else
  echo "== 2/4 features đã có: ${FEATS} =="
fi

echo "== 3/4 VALIDATION: dec (Rust) vs PyTorch =="
cargo run --release -p voice-models --example verify_dec | tail -4

echo "== 4/4 TRAINING: adapter (SEG=${SEG} STEPS=${STEPS} LR=${LR}) =="
SEG="${SEG}" STEPS="${STEPS}" LR="${LR}" \
  cargo run --release -p voice-train --example train_adapter

echo "✔ xong: test_output.wav + sources/features/adapter.bpk"
