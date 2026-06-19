#!/usr/bin/env bash
set -u

MODEL_DIR="${MODEL_DIR:-../models/Qwen2.5-Omni-3B}"
warnings=0

ok() { printf '[ok] %s\n' "$1"; }
warn() { printf '[warn] %s\n' "$1"; warnings=$((warnings + 1)); }

echo "== Qwen2.5-Omni-3B local model path check =="
echo "MODEL_DIR=${MODEL_DIR}"
echo

if [ -d "$MODEL_DIR" ]; then
  ok "model directory exists"
else
  warn "model directory missing; set MODEL_DIR to the local Qwen2.5-Omni-3B checkpoint path"
  exit 0
fi

for file in \
  config.json \
  model.safetensors.index.json \
  model-00001-of-00003.safetensors \
  model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors; do
  if [ -f "$MODEL_DIR/$file" ]; then
    ok "$file exists"
  else
    warn "$file missing"
  fi
done

tokenizer_found=0
for file in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  if [ -f "$MODEL_DIR/$file" ]; then
    ok "tokenizer file found: $file"
    tokenizer_found=1
  fi
done
[ "$tokenizer_found" -eq 1 ] || warn "no tokenizer file found"

processor_found=0
for file in preprocessor_config.json processor_config.json video_preprocessor.json image_processor_config.json; do
  if [ -f "$MODEL_DIR/$file" ]; then
    ok "processor/preprocessor file found: $file"
    processor_found=1
  fi
done
[ "$processor_found" -eq 1 ] || warn "no processor/preprocessor config found"

echo
du -sh "$MODEL_DIR" 2>/dev/null || true
echo "warnings: $warnings"
