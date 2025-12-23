# BirdCLEF+ 2025 - AI Coding Instructions

This is a competition-winning bird sound classification system using PyTorch Lightning and HDF5-optimized audio processing.

## Architecture Overview

**Core Pipeline**: Audio → HDF5 preprocessing → CNN with attention → Multi-label classification
- **Models**: `WaveCNNAttenClasifier` with `tf_efficientnetv2_s_in21k` or `eca_nfnet_l0` backbones
- **Audio Processing**: 32kHz → Mel spectrograms (128 mels, 2048 FFT, 512 hop) → CNN features
- **Training**: 5-fold CV with PyTorch Lightning, pseudo-labeling, mixup augmentation
- **Inference**: Ensemble averaging across folds, ONNX/OpenVINO export for production

## Key File Structure

```
code_base/
├── datasets/wave_dataset.py       # Core dataset classes with HDF5 support
├── models/wave_clasifier.py       # Main CNN+attention architecture  
├── train_functions/train_lightning.py  # Lightning training wrapper
├── inefernce/inference_class.py   # BirdsInference for validation/test
├── augmentations/                 # Audio & spectrogram augmentations
└── utils/                        # Audio loading, metrics, ONNX utils

train_configs/                     # Training configurations per model
inference_configs/                 # Inference configurations per model
scripts/
├── main_train.py                 # CV training entry point
├── main_inference_and_compile.py # Validation & ONNX export
└── precompute_features.py        # Audio→HDF5 conversion
```

## Critical Development Patterns

### Configuration-Driven Design
All experiments use Python config files with `CONFIG` dicts:
```python
# train_configs/selected_ebs.py
CONFIG = {
    "exp_name": "tf_efficientnetv2_s_in21k_...",
    "train_function": lightning_training,
    "nn_model_class": WaveCNNAttenClasifier,
    "nn_model_config": {...},
    "train_function_args": {...}
}
```
- Training configs in `train_configs/`, inference configs in `inference_configs/`
- Config names must match between training/inference
- All hyperparameters centralized in these files

### HDF5-First Audio Processing
Audio files are preprocessed to HDF5 for training efficiency:
```bash
python scripts/precompute_features.py data/train_audio data/train_features --n_cores 8

# Partial conversion (useful for large datasets with limited disk space)
python scripts/precompute_features.py data/train_audio data/train_features --n_cores 8 --only_biggest --leave_gb 100
```
- `WaveDataset` supports both raw audio and HDF5 (`use_h5py=True`)
- **Mixed datasets supported**: When `use_h5py=True`, datasets automatically detect per-file whether HDF5 exists, falling back to raw audio
- HDF5 provides ~10x faster loading during training
- Use `--only_biggest` flag to convert largest files first when disk space is limited
- Dataset prints composition: `"Dataset composition: X HDF5 files, Y raw audio files"`

### Dataset Architecture
Two main classes:
- `WaveDataset`: Training with random crops, mixup, augmentations
- `WaveAllFileDataset`: Validation/inference with systematic sliding windows

Key parameters:
- `segment_len=5.0`: 5-second audio clips
- `sample_rate=32000`: Fixed sample rate
- `late_normalize=True`: Normalize after augmentations

### Training Workflow
```bash
# 1. Precompute HDF5 features
python scripts/precompute_features.py data/audio data/features --n_cores 8

# 2. Train with config
CUDA_VISIBLE_DEVICES="0" python scripts/main_train.py train_configs/selected_ebs.py

# 3. Generate predictions & export
CUDA_VISIBLE_DEVICES="0" python scripts/main_inference_and_compile.py inference_configs/selected_ebs.py
```

### Model Ensembling
- 5-fold cross-validation standard
- Models stored as `logdirs/{exp_name}/fold_{i}/checkpoints/`
- `BirdsInference` class handles multi-fold ensemble averaging
- Supports ONNX and OpenVINO export for deployment

## Key Implementation Details

### Audio Augmentation Strategy
1. **Early augmentation**: On raw waveforms (limited use)
2. **Late augmentation**: Background noise mixing, before normalization
3. **Spectral augmentation**: Freq/time masking, power scaling on spectrograms
4. **Mixup**: Sample blending with configurable alpha

### Pseudo-Labeling System
- Soundscape data gets pseudo-labels from trained models
- `soundscape_pseudo_config` controls confidence thresholds
- Integrated into training via `WaveDataset` sampling logic

### Memory Management
- Audio caching with `precompute=True` for small datasets
- `use_audio_cache=True` for inference to avoid reloading
- Explicit garbage collection in dataset classes

## Common Commands

```bash
# Full pipeline (download→preprocess→train→export)
bash rock_that_bird.sh "0"

# Train specific model
WANDB_MODE="offline" CUDA_VISIBLE_DEVICES="0" python scripts/main_train.py train_configs/selected_ebs.py

# Convert to production format
CUDA_VISIBLE_DEVICES="0" python scripts/main_inference_and_compile.py inference_configs/selected_ebs.py

# Precompute features for new data
python scripts/precompute_features.py /path/to/audio /path/to/hdf5 --n_cores 8 --use_torchaudio
```

## Dependencies & Environment
- Poetry for dependency management (`poetry install`)
- CUDA required for training (RTX 4090 recommended)
- ~600GB disk space for full dataset
- Key packages: `torch`, `lightning`, `h5py`, `librosa`, `timm`

When modifying this codebase:
1. Always test with small datasets first (`debug=True`)
2. Ensure config name consistency between train/inference
3. Use HDF5 preprocessing for any new audio data
4. Follow the configuration-driven pattern for new experiments
5. Test ONNX export if modifying model architecture
