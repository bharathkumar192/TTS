#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert JSON data to Coqui TTS metadata format for XTTS fine-tuning
"""

import os
import json
import argparse
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm


def convert_json_to_metadata(json_file, audio_dir, output_dir, sample_rate=24000, val_ratio=0.15):
    """
    Convert JSON file to Coqui TTS metadata format and organize audio files
    
    Args:
        json_file: Path to JSON file containing sentences
        audio_dir: Path to directory containing audio files
        output_dir: Path to output directory
        sample_rate: Target sample rate for audio files
        val_ratio: Ratio of validation set size
    """
    print(f"Converting {json_file} to TTS metadata format...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create metadata entries
    metadata = []
    missing_files = []
    
    for entry in tqdm(data, desc="Processing files"):
        sentence_id = entry["sentence_id"]
        hindi_sentence = entry["hindi_sentence"]
        
        # Source audio file
        audio_file = os.path.join(audio_dir, f"{sentence_id}.wav")
        
        # Target audio file
        target_audio_file = os.path.join(output_dir, "wavs", f"{sentence_id}.wav")
        
        # Check if audio file exists
        if os.path.exists(audio_file):
            # Process audio file (resample if needed)
            try:
                y, sr = librosa.load(audio_file, sr=None)
                if sr != sample_rate:
                    print(f"Resampling {audio_file} from {sr} to {sample_rate} Hz")
                    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                
                # Save to target location
                sf.write(target_audio_file, y, sample_rate)
                
                # Format: file_id|text
                metadata.append(f"{sentence_id}|{hindi_sentence}")
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                missing_files.append(sentence_id)
        else:
            missing_files.append(sentence_id)
    
    # Split into train and validation sets
    import random
    random.shuffle(metadata)
    
    val_size = int(len(metadata) * val_ratio)
    train_metadata = metadata[val_size:]
    val_metadata = metadata[:val_size]
    
    # Write metadata files
    train_file = os.path.join(output_dir, "metadata_train.csv")
    val_file = os.path.join(output_dir, "metadata_val.csv")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_metadata))
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_metadata))
    
    # Create dataset config file
    dataset_config = {
        "meta_file_train": "metadata_train.csv",
        "meta_file_val": "metadata_val.csv",
        "path": "wavs",
        "language": "hi"
    }
    
    dataset_config_path = os.path.join(output_dir, "dataset_config.json")
    with open(dataset_config_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"Created train metadata with {len(train_metadata)} entries at {train_file}")
    print(f"Created validation metadata with {len(val_metadata)} entries at {val_file}")
    print(f"Created dataset config at {dataset_config_path}")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} audio files are missing or couldn't be processed:")
        for file_id in missing_files[:10]:
            print(f"  - {file_id}.wav")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return train_file, val_file, dataset_config_path


def create_training_config(output_dir, batch_size=8, epochs=10, learning_rate=1e-5, early_stop_patience=3):
    """
    Create training configuration file for XTTS fine-tuning
    
    Args:
        output_dir: Path to output directory
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        early_stop_patience: Number of epochs to wait for early stopping
    """
    config = {
        "run_name": "xtts_hindi_finetune",
        "output_path": os.path.join(output_dir, "output"),
        "batch_size": batch_size,
        "eval_batch_size": batch_size,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 2,
        "epochs": epochs,
        "lr": learning_rate,
        "optimizer": "AdamW",
        "optimizer_params": {
            "weight_decay": 1e-6
        },
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {
            "T_max": epochs
        },
        "grad_clip": 5.0,
        "early_stopping_patience": early_stop_patience,
        "mixed_precision": True,
        "save_n_checkpoints": 3,
        "save_best_after": 1000,
        "print_step": 50,
        "language": "hi"
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created training config at {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to Coqui TTS metadata format")
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON file containing sentences")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate for audio files")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation set size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Number of epochs to wait for early stopping")
    args = parser.parse_args()
    
    # Convert JSON to metadata and organize audio files
    train_file, val_file, dataset_config = convert_json_to_metadata(
        args.json_file,
        args.audio_dir,
        args.output_dir,
        args.sample_rate,
        args.val_ratio
    )
    
    # Create training config
    config_path = create_training_config(
        args.output_dir,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.early_stop_patience
    )
    
    print("\nData preparation complete!")
    print("\nTo start fine-tuning, run:")
    print(f"tts finetune --config_path {config_path} \\\n  --model_path tts_models/multilingual/multi-dataset/xtts_v2 \\\n  --dataset_config_path {dataset_config} \\\n  --use_cuda true")
    
    # Multi-GPU command
    print("\nFor multi-GPU training, run:")
    print(f"tts finetune --config_path {config_path} \\\n  --model_path tts_models/multilingual/multi-dataset/xtts_v2 \\\n  --dataset_config_path {dataset_config} \\\n  --use_cuda true \\\n  --strategy ddp \\\n  --num_gpus 2")


if __name__ == "__main__":
    main()