#!/usr/bin/env bash
# Tải BINARY đã build sẵn (cargo-dist, Linux x86_64) từ GitHub Release.
# Dùng trên Colab để huấn luyện/validate mà KHÔNG cần cài Rust hay biên dịch.
#
#   ./scripts/fetch_bin.sh              # tag mặc định (xem TAG bên dưới)
#   TAG=v0.1.0 ./scripts/fetch_bin.sh   # chọn tag khác
#
# Giải nén vào ./bin/{voice}.
set -euo pipefail

REPO="${REPO:-hungpham10/voice-rs}"
TAG="${TAG:-v2.2.14}"
TARGET="x86_64-unknown-linux-gnu"

mkdir -p bin

fetch_bin() { # <package> <binary-name>
  local pkg="$1" bin="$2"
  local asset="${pkg}-${TARGET}.tar.xz"
  local url="https://github.com/${REPO}/releases/download/${TAG}/${asset}"
  local tmp="${TMPDIR:-/tmp}/${asset}"
  local stage
  stage="$(mktemp -d)"

  echo "→ ${bin}  (${asset})"
  curl -L --fail --progress-bar -o "${tmp}" "${url}"
  tar -xJf "${tmp}" -C "${stage}"
  # cargo-dist bọc mọi thứ trong thư mục <pkg>-<target>/ (binary + LICENSE/README)
  # → lấy binary từ trong đó.
  mv "${stage}/${pkg}-${TARGET}/${bin}" bin/
  chmod +x "bin/${bin}"
  rm -rf "${stage}" "${tmp}"
}

fetch_bin voice-train voice

echo "✔ binary sẵn sàng trong ./bin/"
ls -lh bin/
