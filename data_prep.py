#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation script for XTTS fine-tuning with Hindi data
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
from pathlib import Path
import shutil


def create_metadata_csv(json_file, audio_dir, output_dir, sample_rate=24000):
    """
    Create metadata.csv from JSON file and audio directory
    
    Args:
        json_file: Path to JSON file containing sentences
        audio_dir: Path to directory containing audio files
        output_dir: Path to output directory for processed data
        sample_rate: Target sample rate for audio
    """
    print(f"Creating metadata from {json_file} with audio from {audio_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    processed_audio_dir = os.path.join(output_dir, "wavs")
    os.makedirs(processed_audio_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create metadata list
    metadata = []
    
    # Process each entry
    for entry in tqdm(data, desc="Processing audio files"):
        sentence_id = entry["sentence_id"]
        sentence = entry["hindi_sentence"]
        
        # Source audio file
        audio_file = os.path.join(audio_dir, f"{sentence_id}.wav")
        
        # Target audio file
        target_audio_file = os.path.join(processed_audio_dir, f"{sentence_id}.wav")
        
        # Check if audio file exists
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found, skipping")
            continue
        
        # Load and resample audio if needed
        try:
            y, sr = librosa.load(audio_file, sr=None)
            if sr != sample_rate:
                print(f"Resampling {audio_file} from {sr} to {sample_rate} Hz")
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            
            # Save to target location
            sf.write(target_audio_file, y, sample_rate)
            
            # Add to metadata
            metadata.append(f"{sentence_id}|{sentence}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Write metadata file
    metadata_file = os.path.join(output_dir, "metadata.csv")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata))
    
    print(f"Created metadata file with {len(metadata)} entries at {metadata_file}")
    return metadata_file


def verify_dataset(metadata_file, audio_dir):
    """
    Verify that all audio files referenced in metadata exist
    
    Args:
        metadata_file: Path to metadata CSV file
        audio_dir: Path to directory containing audio files
    """
    print(f"Verifying dataset at {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    missing_files = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2:
            continue
        
        file_id = parts[0]
        audio_file = os.path.join(audio_dir, f"{file_id}.wav")
        
        if not os.path.exists(audio_file):
            missing_files.append(file_id)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} audio files are missing: {missing_files[:5]}...")
    else:
        print("All audio files referenced in metadata exist.")


def create_config_json(output_dir, base_model_path, batch_size=8, grad_accum_steps=4, 
                      epochs=10, learning_rate=1e-5, early_stop_patience=3):
    """
    Create config.json for XTTS fine-tuning
    
    Args:
        output_dir: Path to output directory
        base_model_path: Path to base XTTS model
        batch_size: Batch size per GPU
        grad_accum_steps: Gradient accumulation steps
        epochs: Number of epochs to train
        learning_rate: Learning rate
        early_stop_patience: Number of epochs to wait before early stopping
    """
    config = {
        "run_name": "xtts_hindi_finetune",
        "output_path": os.path.join(output_dir, "runs"),
        "data": {
            "path": os.path.join(output_dir, "metadata.csv"),
            "audio_dir": os.path.join(output_dir, "wavs")
        },
        "audio": {
            "sample_rate": 24000,
            "win_length": 1024,
            "hop_length": 256
        },
        "trainer": {
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "epochs": epochs,
            "lr": learning_rate,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "early_stop_patience": early_stop_patience
        },
        "model": {
            "type": "xtts",
            "use_d_vector_file": False,
            "base_model_path": base_model_path
        },
        "eval": {
            "sample_size": 100, 
            "metrics": ["wer", "speaker_cosine"]
        }
    }
    
    config_path = os.path.join(output_dir, "xtts_hindi_finetune.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file at {config_path}")
    return config_path


def split_train_val_data(metadata_file, output_dir, val_ratio=0.15):
    """
    Split metadata into training and validation sets
    
    Args:
        metadata_file: Path to metadata CSV file
        output_dir: Path to output directory
        val_ratio: Ratio of validation data
    """
    print(f"Splitting metadata into training and validation sets with val_ratio={val_ratio}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle lines
    import random
    random.shuffle(lines)
    
    # Split into train and val
    val_size = int(len(lines) * val_ratio)
    train_lines = lines[val_size:]
    val_lines = lines[:val_size]
    
    # Write train metadata
    train_metadata_file = os.path.join(output_dir, "metadata_train.csv")
    with open(train_metadata_file, 'w', encoding='utf-8') as f:
        f.write(''.join(train_lines))
    
    # Write val metadata
    val_metadata_file = os.path.join(output_dir, "metadata_val.csv")
    with open(val_metadata_file, 'w', encoding='utf-8') as f:
        f.write(''.join(val_lines))
    
    print(f"Created training metadata with {len(train_lines)} entries at {train_metadata_file}")
    print(f"Created validation metadata with {len(val_lines)} entries at {val_metadata_file}")
    return train_metadata_file, val_metadata_file


def main():
    parser = argparse.ArgumentParser(description="Prepare data for XTTS fine-tuning")
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON file containing sentences")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for processed data")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base XTTS model checkpoint")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Number of epochs to wait before early stopping")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation data")
    args = parser.parse_args()
    
    # Create metadata CSV
    metadata_file = create_metadata_csv(args.json_file, args.audio_dir, args.output_dir, args.sample_rate)
    
    # Verify dataset
    verify_dataset(metadata_file, os.path.join(args.output_dir, "wavs"))
    
    # Split train/val data
    train_metadata_file, val_metadata_file = split_train_val_data(metadata_file, args.output_dir, args.val_ratio)
    
    # Create config JSON
    config_path = create_config_json(
        args.output_dir, 
        args.base_model_path,
        args.batch_size,
        args.grad_accum_steps,
        args.epochs,
        args.learning_rate,
        args.early_stop_patience
    )
    
    print("\nData preparation complete! Next steps:")
    print(f"1. Run training with: tts finetune --config {config_path} --use_cuda --strategy ddp --num_workers 4")


if __name__ == "__main__":
    main()