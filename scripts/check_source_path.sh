#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_REPO="${PROJECT_ROOT}/third_party/EfficientAI"
SOURCE_DIR="${SOURCE_REPO}/masquant"

echo "== MASQuant source path check =="
echo "SOURCE_REPO=${SOURCE_REPO}"
echo "SOURCE_DIR=${SOURCE_DIR}"
echo

if [ -d "$SOURCE_REPO" ]; then
  echo "[ok] EfficientAI repo exists"
  if command -v git >/dev/null 2>&1 && [ -d "$SOURCE_REPO/.git" ]; then
    git -C "$SOURCE_REPO" remote -v || true
  fi
else
  echo "[warn] EfficientAI repo is missing"
  echo "Run:"
  echo "  cd \"$PROJECT_ROOT/third_party\""
  echo "  git clone https://github.com/alibaba/EfficientAI.git"
fi

if [ -d "$SOURCE_DIR" ]; then
  echo "[ok] MASQuant source directory exists"
else
  echo "[warn] MASQuant source directory missing: $SOURCE_DIR"
fi
