#!/usr/bin/env python3
"""
Test script to verify mixed HDF5/raw audio dataset handling.

This script creates a minimal test case with both HDF5 and raw audio files
to ensure the WaveDataset handles mixed formats correctly.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import h5py
import librosa
import soundfile as sf
from code_base.datasets import WaveDataset
import json

def create_test_data():
    """Create temporary test data with both HDF5 and raw audio files."""
    temp_dir = tempfile.mkdtemp()
    print(f"Creating test data in: {temp_dir}")
    
    # Create directory structure
    audio_dir = os.path.join(temp_dir, "audio")
    features_dir = os.path.join(temp_dir, "features")
    
    os.makedirs(os.path.join(audio_dir, "bird1"), exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "bird2"), exist_ok=True)
    os.makedirs(os.path.join(features_dir, "bird1"), exist_ok=True)
    
    # Create sample audio
    sr = 32000
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create 3 audio files: 2 will have HDF5, 1 will not
    files_info = []
    
    # File 1: HDF5 + raw audio (bird1)
    audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wav_path1 = os.path.join(audio_dir, "bird1", "sample1.wav")
    hdf5_path1 = os.path.join(features_dir, "bird1", "sample1.hdf5")
    
    sf.write(wav_path1, audio1, sr)
    with h5py.File(hdf5_path1, "w") as f:
        f.create_dataset("au", data=audio1)
        f.create_dataset("sr", data=sr)
        f.create_dataset("do_normalize", data=0)
    
    files_info.append({
        "filename": "bird1/sample1.wav",
        "primary_label": "bird1",
        "secondary_labels": "[]",
        "duration_s": duration
    })
    
    # File 2: HDF5 + raw audio (bird1)
    audio2 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    wav_path2 = os.path.join(audio_dir, "bird1", "sample2.wav")
    hdf5_path2 = os.path.join(features_dir, "bird1", "sample2.hdf5")
    
    sf.write(wav_path2, audio2, sr)
    with h5py.File(hdf5_path2, "w") as f:
        f.create_dataset("au", data=audio2)
        f.create_dataset("sr", data=sr)
        f.create_dataset("do_normalize", data=0)
    
    files_info.append({
        "filename": "bird1/sample2.wav",
        "primary_label": "bird1",
        "secondary_labels": "[]",
        "duration_s": duration
    })
    
    # File 3: ONLY raw audio, no HDF5 (bird2)
    audio3 = np.sin(2 * np.pi * 1320 * t).astype(np.float32)
    wav_path3 = os.path.join(audio_dir, "bird2", "sample3.wav")
    # Note: We deliberately don't create HDF5 for this one
    
    sf.write(wav_path3, audio3, sr)
    
    files_info.append({
        "filename": "bird2/sample3.wav",
        "primary_label": "bird2",
        "secondary_labels": "[]",
        "duration_s": duration
    })
    
    # Create CSV
    df = pd.DataFrame(files_info)
    csv_path = os.path.join(temp_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    
    # Create label mapping
    label_mapping = {"bird1": 0, "bird2": 1}
    mapping_path = os.path.join(temp_dir, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f)
    
    return temp_dir, audio_dir, features_dir, csv_path, mapping_path


def test_mixed_dataset():
    """Test that WaveDataset handles mixed HDF5/raw audio correctly."""
    
    print("=" * 60)
    print("Testing Mixed HDF5/Raw Audio Dataset Support")
    print("=" * 60)
    
    # Create test data
    temp_dir, audio_dir, features_dir, csv_path, mapping_path = create_test_data()
    
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        print("\n1. Creating WaveDataset with use_h5py=True")
        print("-" * 60)
        
        # Create dataset with use_h5py=True (should handle mixed formats)
        dataset = WaveDataset(
            df=df,
            root=audio_dir,
            label_str2int_mapping_path=mapping_path,
            replace_pathes=("audio", "features"),
            segment_len=5.0,
            sample_rate=32000,
            use_h5py=True,
            late_normalize=True,
            check_all_files_exist=True,
            debug=False
        )
        
        print(f"\n✓ Dataset created successfully with {len(dataset)} samples")
        
        # Check the composition
        if hasattr(dataset.df, 'is_hdf5'):
            hdf5_count = dataset.df['is_hdf5'].sum()
            raw_count = (~dataset.df['is_hdf5']).sum()
            print(f"✓ Dataset composition detected:")
            print(f"  - HDF5 files: {hdf5_count}")
            print(f"  - Raw audio files: {raw_count}")
            
            assert hdf5_count == 2, f"Expected 2 HDF5 files, got {hdf5_count}"
            assert raw_count == 1, f"Expected 1 raw audio file, got {raw_count}"
        else:
            print("✗ Warning: 'is_hdf5' column not found in dataset")
        
        print("\n2. Testing sample loading from mixed dataset")
        print("-" * 60)
        
        # Test loading samples
        for i in range(len(dataset)):
            wave, target = dataset[i]
            is_hdf5 = dataset.df.iloc[i]['is_hdf5'] if 'is_hdf5' in dataset.df.columns else None
            filename = dataset.df.iloc[i]['filename_with_root']
            
            print(f"\n  Sample {i}:")
            print(f"    File: {os.path.basename(filename)}")
            print(f"    Format: {'HDF5' if is_hdf5 else 'Raw audio'}")
            print(f"    Wave shape: {wave.shape}")
            print(f"    Target shape: {target.shape}")
            
            assert wave.shape[0] == 32000 * 5, f"Expected 160000 samples, got {wave.shape[0]}"
            assert target.shape[0] == 2, f"Expected 2 classes, got {target.shape[0]}"
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nMixed HDF5/raw audio dataset support is working correctly.")
        print("The dataset can now handle partial HDF5 conversion.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up test directory: {temp_dir}")


if __name__ == "__main__":
    test_mixed_dataset()
