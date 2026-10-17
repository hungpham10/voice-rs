#!/usr/bin/env bash
# Tải bundle model đã đóng gói từ GitHub Release và giải nén vào repo.
# Dùng trên Colab (hoặc máy sạch) để có `sources/` mà không cần tự export.
#
#   ./scripts/fetch_assets.sh [tag]    # mặc định assets-v1
set -euo pipefail

REPO="${REPO:-hungpham10/voice-rs}"
TAG="${1:-assets-v1}"
ASSET="voice-assets.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
TMP="${TMPDIR:-/tmp}/${ASSET}"

echo "→ tải ${URL}"
curl -L --fail --progress-bar -o "${TMP}" "${URL}"

echo "→ giải nén vào $(pwd)"
tar -xzf "${TMP}" -C .

# `train_adapter` mặc định đọc wav ở `sources/`, nhưng bundle lưu ở `samples/`.
if [ ! -f sources/422-122949-0031.wav ]; then
  mkdir -p sources
  cp samples/422-122949-0031.wav sources/422-122949-0031.wav
fi

echo "✔ sources/ sẵn sàng:"
ls -lh sources/onnx_hubert_new/model.onnx sources/vits_dec/dec/conv_pre.weight.npy \
       sources/features/422-122949-0031.npy 2>/dev/null || true
