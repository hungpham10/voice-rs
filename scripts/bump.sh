#!/usr/bin/env bash
set -euo pipefail

# bump.sh — bump workspace version, commit, push (KHÔNG đánh tag, KHÔNG release).
#
# Usage:
#   ./bump.sh           # patch (2.2.14 -> 2.2.15)
#   ./bump.sh minor     # 2.2.14 -> 2.3.0
#   ./bump.sh major     # 2.2.14 -> 3.0.0
#   ./bump.sh 2.3.1     # đặt version cụ thể


ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CARGO_TOML="Cargo.toml"
CURRENT_VERSION=$(grep -E '^version = "[0-9]+\.[0-9]+\.[0-9]+"' "$CARGO_TOML" | head -1 | sed -E 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)"/\1/')

if [[ -z "$CURRENT_VERSION" ]]; then
  echo "❌ Không tìm thấy version trong $CARGO_TOML" >&2
  exit 1
fi

MODE="${1:-patch}"

bump_part() {
  local v="$1" part="$2"
  local major minor patch
  major=$(echo "$v" | cut -d. -f1)
  minor=$(echo "$v" | cut -d. -f2)
  patch=$(echo "$v" | cut -d. -f3)
  case "$part" in
    major) echo "$((major + 1)).0.0" ;;
    minor) echo "$major.$((minor + 1)).0" ;;
    patch|*) echo "$major.$minor.$((patch + 1))" ;;
  esac
}

case "$MODE" in
  major|minor|patch) NEW_VERSION=$(bump_part "$CURRENT_VERSION" "$MODE") ;;
  *)
    if [[ ! "$MODE" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "❌ Tham số không hợp lệ: $MODE (dùng major|minor|patch|X.Y.Z)" >&2
      exit 1
    fi
    NEW_VERSION="$MODE"
    ;;
esac

if [[ "$NEW_VERSION" == "$CURRENT_VERSION" ]]; then
  echo "ℹ️  Đã ở version $CURRENT_VERSION, không có gì để làm."
  exit 0
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "❌ Working tree chưa sạch. Hãy commit/stash các thay đổi trước khi bump." >&2
  git status --porcelain >&2
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" == "HEAD" ]]; then
  echo "❌ Đang ở detached HEAD. Hãy checkout một branch trước khi bump." >&2
  exit 1
fi

echo "Bump: $CURRENT_VERSION → $NEW_VERSION (branch: $BRANCH)"

NEW_VERSION_PERL="$NEW_VERSION" perl -pi -e 's/^version = "[0-9]+.[0-9]+.[0-9]+"/version = "$ENV{NEW_VERSION_PERL}"/' "$CARGO_TOML"

cargo update --workspace --quiet

git add "$CARGO_TOML" Cargo.lock
if [[ -n "$(git diff --cached --name-only)" ]]; then
  git commit -S -m "chore: bump version to $NEW_VERSION"
  git push origin "$BRANCH"
  echo "✅ Đã commit & push version $NEW_VERSION lên branch $BRANCH (chưa tag)."
else
  echo "ℹ️  Không có thay đổi để commit."
fi

