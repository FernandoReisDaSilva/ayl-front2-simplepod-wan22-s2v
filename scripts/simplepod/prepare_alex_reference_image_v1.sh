#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

SRC="$HOME/Downloads/Alex para Wan V3.png"
DEST_DIR="data/simplepod_wan22_s2v/inputs/alex_en_wan22_s2v_720_v1/reference"
DEST="$DEST_DIR/alex_en_reference.png"

if [ ! -f "$SRC" ]; then
  echo "ERROR: source image not found:"
  echo "$SRC"
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"

echo "Copied Alex reference image:"
echo "from: $SRC"
echo "to:   $DEST"

ls -lh "$DEST"
