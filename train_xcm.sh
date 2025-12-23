#!/bin/bash

echo "GPU $1 will be used for training"

# -----------------------------
# Precompute features
# -----------------------------
# python scripts/precompute_features.py /workspace/birdsongs/data/xcm/train /workspace/birdsongs/data/xcm_hdf5/train --only_biggest

# -----------------------------
# Start training
# -----------------------------
CUDA_VISIBLE_DEVICES="$1" python scripts/main_train.py train_configs/eca_xcm.py
