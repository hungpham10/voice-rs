#!/usr/bin/env bash
# Tải bundle MODEL đã đóng gói từ GitHub Release và giải nén vào repo.
# Dùng trên Colab (hoặc máy sạch) để có `sources/` mà không cần tự export.
#
#   ./scripts/fetch_assets.sh [tag]           # bundle chính (ContentVec + VITS dec)
#   ./scripts/fetch_assets.sh --f0 [tag]      # lấy thêm rmvpe.onnx cho voice-f0
set -euo pipefail

REPO="${REPO:-hungpham10/voice-rs}"
WANT_F0=0
if [ "${1:-}" = "--f0" ]; then WANT_F0=1; shift; fi
TAG="${1:-assets-v1}"

fetch() { # <asset-name>
  local asset="$1"
  local url="https://github.com/${REPO}/releases/download/${TAG}/${asset}"
  local tmp="${TMPDIR:-/tmp}/${asset}"
  echo "→ tải ${url}"
  curl -L --fail --progress-bar -o "${tmp}" "${url}"
  tar -xzf "${tmp}" -C .
  # Bundle kèm SHA256SUMS; kiểm từng file (sha256sum chuẩn trên Linux/Colab).
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c SHA256SUMS
  else
    shasum -a 256 -c SHA256SUMS
  fi
  rm -f "${tmp}" SHA256SUMS
}

fetch voice-assets.tar.gz
[ "${WANT_F0}" = "1" ] && fetch f0-assets.tar.gz || true

# `train_adapter` mặc định đọc wav ở `sources/`; wav mẫu nằm trong repo.
if [ ! -f sources/422-122949-0031.wav ] && [ -f samples/422-122949-0031.wav ]; then
  mkdir -p sources
  cp samples/422-122949-0031.wav sources/422-122949-0031.wav
fi

echo "✔ sources/ sẵn sàng:"
ls -lh sources/onnx_hubert_new/model.onnx sources/vits_dec/dec/conv_pre.weight.npy 2>/dev/null || true
[ "${WANT_F0}" = "1" ] && ls -lh sources/rmvpe.onnx 2>/dev/null || true
