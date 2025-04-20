#!/bin/bash
# Script to run the full XTTS fine-tuning pipeline using Coqui TTS built-in tools

# Default values
JSON_FILE="final.json"
AUDIO_DIR="final_audio_24hz"
OUTPUT_DIR="xtts_hindi_finetune"
NUM_GPUS=1
BATCH_SIZE=8
EPOCHS=10
LEARNING_RATE=1e-5
EARLY_STOP_PATIENCE=3
VAL_RATIO=0.15
SAMPLE_RATE=24000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --json_file)
      JSON_FILE="$2"
      shift
      shift
      ;;
    --audio_dir)
      AUDIO_DIR="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift
      shift
      ;;
    --early_stop_patience)
      EARLY_STOP_PATIENCE="$2"
      shift
      shift
      ;;
    --val_ratio)
      VAL_RATIO="$2"
      shift
      shift
      ;;
    --sample_rate)
      SAMPLE_RATE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "========== XTTS Hindi Fine-tuning =========="
echo "JSON File: $JSON_FILE"
echo "Audio Directory: $AUDIO_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Early Stop Patience: $EARLY_STOP_PATIENCE"
echo "Validation Ratio: $VAL_RATIO"
echo "Sample Rate: $SAMPLE_RATE"
echo "==========================================="

# Install dependencies if needed
echo "Checking and installing dependencies..."
pip install -q TTS==0.22.0 resemblyzer openai-whisper matplotlib pandas librosa soundfile transformers torch-bin google-api-python-client

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Download XTTS model - store the base path
echo "Downloading XTTS base model..."
# Use a simpler approach for getting the model path
python -c "from TTS.utils.manage import ModelManager; ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2')"
# Just use the standard path rather than trying to parse the output
BASE_MODEL_PATH="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
echo "Base model path: $BASE_MODEL_PATH"

# Step 2: Convert JSON to Coqui format and prepare data files
echo "Converting JSON data to Coqui TTS format..."
# Create metadata.csv directly
python -c "
import json
import os
import librosa
import soundfile as sf
from tqdm import tqdm

# Load JSON data
with open('$JSON_FILE', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create output directories
os.makedirs('$OUTPUT_DIR/wavs', exist_ok=True)

# Process data
train_metadata = []
val_metadata = []
all_entries = []

# First, collect all valid entries
for entry in data:
    sentence_id = entry['sentence_id']
    hindi_sentence = entry['hindi_sentence']
    
    audio_file = os.path.join('$AUDIO_DIR', f'{sentence_id}.wav')
    target_audio_file = os.path.join('$OUTPUT_DIR/wavs', f'{sentence_id}.wav')
    
    if os.path.exists(audio_file):
        try:
            # Process audio
            y, sr = librosa.load(audio_file, sr=None)
            if sr != $SAMPLE_RATE:
                print(f'Resampling {audio_file} from {sr} to $SAMPLE_RATE Hz')
                y = librosa.resample(y, orig_sr=sr, target_sr=$SAMPLE_RATE)
            
            # Save to target location
            sf.write(target_audio_file, y, $SAMPLE_RATE)
            
            # Add to entries
            all_entries.append(f'{sentence_id}|{hindi_sentence}')
            
        except Exception as e:
            print(f'Error processing {audio_file}: {e}')
    else:
        print(f'Warning: Audio file {audio_file} not found')

# Shuffle and split data
import random
random.shuffle(all_entries)

val_size = int(len(all_entries) * $VAL_RATIO)
train_metadata = all_entries[val_size:]
val_metadata = all_entries[:val_size]

# Write metadata files
with open('$OUTPUT_DIR/metadata_train.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_metadata))

with open('$OUTPUT_DIR/metadata_val.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_metadata))

print(f'Created training metadata with {len(train_metadata)} entries')
print(f'Created validation metadata with {len(val_metadata)} entries')
"

# Create dataset config
echo "Creating dataset config..."
cat > "$OUTPUT_DIR/dataset_config.json" << EOL
{
  "meta_file_train": "metadata_train.csv",
  "meta_file_val": "metadata_val.csv",
  "path": "wavs",
  "language": "hi"
}
EOL

# Create training config
echo "Creating training config..."
cat > "$OUTPUT_DIR/config.json" << EOL
{
  "run_name": "xtts_hindi_finetune",
  "output_path": "${OUTPUT_DIR}/output",
  "batch_size": ${BATCH_SIZE},
  "eval_batch_size": ${BATCH_SIZE},
  "num_loader_workers": 4,
  "num_eval_loader_workers": 2,
  "epochs": ${EPOCHS},
  "lr": ${LEARNING_RATE},
  "optimizer": "AdamW",
  "optimizer_params": {
    "weight_decay": 1e-6
  },
  "scheduler": "CosineAnnealingLR",
  "scheduler_params": {
    "T_max": ${EPOCHS}
  },
  "grad_clip": 5.0,
  "early_stopping_patience": ${EARLY_STOP_PATIENCE},
  "mixed_precision": true,
  "save_n_checkpoints": 3,
  "save_best_after": 1000,
  "print_step": 50,
  "language": "hi"
}
EOL

# Step 3: Run fine-tuning
echo "Starting XTTS fine-tuning..."

# Determine command based on number of GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
  echo "Using multi-GPU training with $NUM_GPUS GPUs..."
  tts finetune \
    --config_path "$OUTPUT_DIR/config.json" \
    --model_path "tts_models/multilingual/multi-dataset/xtts_v2" \
    --dataset_config_path "$OUTPUT_DIR/dataset_config.json" \
    --use_cuda true \
    --strategy ddp \
    --num_gpus "$NUM_GPUS"
else
  echo "Using single GPU training..."
  tts finetune \
    --config_path "$OUTPUT_DIR/config.json" \
    --model_path "tts_models/multilingual/multi-dataset/xtts_v2" \
    --dataset_config_path "$OUTPUT_DIR/dataset_config.json" \
    --use_cuda true
fi

# Step 4: Create a test script
echo "Creating test script..."
cat > "$OUTPUT_DIR/test_model.py" << EOL
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for fine-tuned XTTS Hindi model
"""

import os
import sys
import torch
import glob
import random
from pathlib import Path
from TTS.api import TTS

def test_model(model_dir, reference_wav=None, output_file="test_output.wav"):
    """
    Test the fine-tuned XTTS model
    
    Args:
        model_dir: Directory containing the fine-tuned model
        reference_wav: Path to reference wav file (if None, will choose randomly)
        output_file: Path to output wav file
    """
    # Find the model files
    checkpoint_dir = os.path.join(model_dir, "output")
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Could not find model directory at {checkpoint_dir}")
        return
    
    # Find best model path
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(model_path):
        # Try to find checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
        if not checkpoint_files:
            print(f"Error: No model checkpoints found in {checkpoint_dir}")
            return
        
        # Use the latest checkpoint
        model_path = sorted(checkpoint_files)[-1]
    
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    # Find reference wav if not provided
    if reference_wav is None:
        wav_files = glob.glob(os.path.join(model_dir, "wavs", "*.wav"))
        if not wav_files:
            print(f"Error: No wav files found in {os.path.join(model_dir, 'wavs')}")
            return
        
        reference_wav = random.choice(wav_files)
    
    print(f"Using model: {model_path}")
    print(f"Using reference wav: {reference_wav}")
    
    # Initialize TTS with fine-tuned model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_path=model_path, config_path=config_path).to(device)
        
        # Test sentences in Hindi
        test_sentences = [
            "नमस्ते, मेरा नाम क्या है?",
            "आप कैसे हैं? मुझे आशा है कि आप अच्छे हैं।",
            "यह एक परीक्षण वाक्य है जिसे हमारे फाइन-ट्यून्ड मॉडल द्वारा उत्पन्न किया जाएगा।"
        ]
        
        # Generate speech for the first test sentence
        print(f"Generating speech for: {test_sentences[0]}")
        tts.tts_to_file(
            text=test_sentences[0],
            speaker_wav=reference_wav,
            language="hi",
            file_path=output_file
        )
        
        print(f"Generated audio saved to {output_file}")
        
        # Optional: Generate speech for all test sentences
        for i, sentence in enumerate(test_sentences):
            output_file_i = f"test_output_{i+1}.wav"
            print(f"Generating speech for: {sentence}")
            tts.tts_to_file(
                text=sentence,
                speaker_wav=reference_wav,
                language="hi",
                file_path=output_file_i
            )
            print(f"Generated audio saved to {output_file_i}")
        
        return True
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    # Directory containing the fine-tuned model
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Reference wav file (optional)
    reference_wav = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Output file (optional)
    output_file = sys.argv[3] if len(sys.argv) > 3 else "test_output.wav"
    
    test_model(model_dir, reference_wav, output_file)
EOL

chmod +x "$OUTPUT_DIR/test_model.py"

echo "Fine-tuning complete!"
echo "Model saved to: $OUTPUT_DIR/output/"
echo ""
echo "To test your model, run:"
echo "python $OUTPUT_DIR/test_model.py $OUTPUT_DIR"
echo ""
echo "For more information on using the model, refer to the Coqui TTS documentation:"
echo "https://tts.readthedocs.io/en/latest/"