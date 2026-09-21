#!/bin/bash

cd /home/csprunger/ComfyUI

.venv/bin/ctq -i models/text_encoders/Qwen3VL/qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors -o models/text_encoders/Qwen3VL/qwen3vl_32b_h3_ultra_uncensored_heretic_nvfp4.safetensors \
  --nvfp4 --qwen_vlm --comfy_quant --save-quant-metadata \
  --low-memory --device cuda --manual-seed 42 --num-iter 4000 --optimizer adamw --verbose NORMAL

.venv/bin/ctq -i models/text_encoders/Qwen3VL/qwen3vl_32b_h3_ultra_uncensored_heretic_generation_tail_50_63_bf16.safetensors -o models/text_encoders/Qwen3VL/qwen3vl_32b_h3_ultra_uncensored_heretic_generation_tail_50_63_nvfp4.safetensors \
  --nvfp4 --qwen_vlm --comfy_quant --save-quant-metadata \
  --low-memory --device cuda --manual-seed 42 --num-iter 4000 --optimizer adamw --verbose NORMAL