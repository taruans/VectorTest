#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/Users/tahsinertan/git/VectorTest"
VENV_DIR="$BASE_DIR/vector_test"
PYTHON_BIN="$VENV_DIR/bin/python"
APP_PATH="$BASE_DIR/app.py"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Hata: Sanal ortam bulunamadı: $VENV_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Hata: Python yürütülebilir dosyası bulunamadı: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$APP_PATH" ]]; then
  echo "Hata: Uygulama dosyası bulunamadı: $APP_PATH" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

exec "$PYTHON_BIN" "$APP_PATH"
