#!/bin/bash
# Download real tokenizer vocab from HuggingFace
set -e

ASSETS_DIR="app/src/main/assets"
HF_BASE="https://huggingface.co/acul3/chatterbox-executorch/resolve/main"

echo "Downloading tokenizer vocab..."
curl -L "${HF_BASE}/grapheme_mtl_merged_expanded_v1.json" \
    -o "${ASSETS_DIR}/grapheme_mtl_merged_expanded_v1.json"

echo "Done! Vocab saved to ${ASSETS_DIR}/"
