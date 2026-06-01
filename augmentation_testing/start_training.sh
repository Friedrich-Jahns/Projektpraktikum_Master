#!/bin/bash
export ROCM_CACHE_DIR=~/.cache/rocm
export MIOPEN_USER_DB_PATH=~/.cache/miopen
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_FIND_MODE=3
export HSA_CACHE_ENABLE=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME=~/.cache/pytorch_tunableop.csv
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Beim allerersten Mal: TUNING=1, danach auf 0 setzen
export PYTORCH_TUNABLEOP_TUNING=0   # nach erstem Lauf

# python train.py
