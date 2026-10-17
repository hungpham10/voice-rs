#!/usr/bin/env bash
# Cài đặt môi trường voice-rs bằng MỘT lệnh: model (sources/) + binary (bin/)
# + phụ thuộc Python cho khâu dữ liệu. Idempotent — chạy lại chỉ tải phần thiếu.
#
#   ./scripts/setup.sh              # mặc định: model + binary Linux + pip deps
#   ./scripts/setup.sh --f0         # thêm rmvpe_f0.onnx (cho voice-f0)
#   ./scripts/setup.sh --no-bin     # máy dev có Rust: bỏ qua tải binary
#   ./scripts/setup.sh --no-deps    # bỏ qua pip install (không đụng Python)
#
# Colab: KHÔNG cần Rust — binary Linux x86_64 tải sẵn từ GitHub Release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

WANT_F0=0
WANT_BIN=1
WANT_DEPS=1
for arg in "$@"; do
  case "$arg" in
    --f0)      WANT_F0=1 ;;
    --no-bin)  WANT_BIN=0 ;;
    --no-deps) WANT_DEPS=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "✗ tham số lạ: $arg (xem --help)" >&2; exit 2 ;;
  esac
done

# Binary phát hành chỉ cho Linux x86_64 (Colab). Trên OS khác thì bỏ qua.
if [ "$WANT_BIN" = "1" ] && [ "$(uname -s)" != "Linux" ]; then
  echo "⚠️  binary chỉ có cho Linux x86_64 ($(uname -s) đây) — tự build bằng cargo."
  WANT_BIN=0
fi

echo "== 1/3 model (sources/) =="
if [ -f sources/vits_ljs_dec/config.json ] && [ -f sources/contentvec/contentvec_768l12.onnx ]; then
  echo "  đã có — bỏ qua (xoá sources/ để tải lại)"
elif [ "$WANT_F0" = "1" ]; then
  ./scripts/fetch_assets.sh --f0
else
  ./scripts/fetch_assets.sh
fi

echo "== 2/3 phụ thuộc Python cho dữ liệu =="
if [ "$WANT_DEPS" = "1" ]; then
  python3 -m pip install -q -r scripts/data/requirements.txt
else
  echo "  bỏ qua (--no-deps)"
fi

echo "== 3/3 binary (bin/) =="
if [ "$WANT_BIN" = "1" ]; then
  ./scripts/fetch_bin.sh
else
  echo "  bỏ qua (--no-bin) — dùng cargo build/run"
fi

cat <<'EOF'

✔ sẵn sàng. Bước tiếp:
  # dataset VIVOS (~1.5 GB) → hint nhanh
  python scripts/data/prepare.py vivos --download --out data/vivos --limit-per-speaker 20
  python scripts/data/split.py data/vivos --val-ratio 0.1

  # smoke overfit 1 câu (nhanh)
  SEG=86 STEPS=250 ./scripts/run_training.sh

  # Luồng 1 — nhiều giọng (VIVOS)
  DATA=data/vivos ./scripts/run_bridge_training.sh
EOF
