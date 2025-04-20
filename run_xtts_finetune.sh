#!/bin/bash
# Minimal script for XTTS fine-tuning

# Default values
JSON_FILE="final.json"
AUDIO_DIR="final_audio_24khz"
OUTPUT_DIR="xtts_hindi_finetune"
NUM_GPUS=1
BATCH_SIZE=8
EPOCHS=10

# Create directories
mkdir -p "$OUTPUT_DIR/wavs"

echo "========== XTTS Hindi Fine-tuning =========="
echo "JSON File: $JSON_FILE"
echo "Audio Directory: $AUDIO_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "==========================================="

# Step 1: Create metadata files manually (avoiding libraries that might cause errors)
echo "Creating metadata files..."

# This uses basic Python without external libraries
python3 -c "
import json
import os
import shutil
import random

# Load JSON data
with open('$JSON_FILE', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process data
all_entries = []
for entry in data:
    sentence_id = entry['sentence_id']
    hindi_sentence = entry['hindi_sentence']
    
    audio_file = os.path.join('$AUDIO_DIR', f'{sentence_id}.wav')
    target_audio_file = os.path.join('$OUTPUT_DIR/wavs', f'{sentence_id}.wav')
    
    if os.path.exists(audio_file):
        # Copy audio file (no resampling, assuming files are already 24kHz)
        shutil.copy(audio_file, target_audio_file)
        
        # Add to entries
        all_entries.append(f'{sentence_id}|{hindi_sentence}')
        print(f'Processed {audio_file}')
    else:
        print(f'Warning: Audio file {audio_file} not found')

# Shuffle and split data
random.shuffle(all_entries)

val_size = int(len(all_entries) * 0.15)  # 15% for validation
train_entries = all_entries[val_size:]
val_entries = all_entries[:val_size]

# Write metadata files
with open('$OUTPUT_DIR/metadata_train.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_entries))

with open('$OUTPUT_DIR/metadata_val.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_entries))

print(f'Created training metadata with {len(train_entries)} entries')
print(f'Created validation metadata with {len(val_entries)} entries')
"

# Step 2: Create dataset config file
echo "Creating dataset config..."
cat > "$OUTPUT_DIR/dataset_config.json" << EOL
{
  "meta_file_train": "metadata_train.csv",
  "meta_file_val": "metadata_val.csv",
  "path": "wavs",
  "language": "hi"
}
EOL

# Step 3: Create training config
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
  "lr": 1e-5,
  "optimizer": "AdamW",
  "optimizer_params": {
    "weight_decay": 1e-6
  },
  "scheduler": "CosineAnnealingLR",
  "scheduler_params": {
    "T_max": ${EPOCHS}
  },
  "grad_clip": 5.0,
  "early_stopping_patience": 3,
  "mixed_precision": true,
  "save_n_checkpoints": 3,
  "save_best_after": 1000,
  "print_step": 50,
  "language": "hi"
}
EOL

# Step 4: Run fine-tuning
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

echo "Fine-tuning complete!"
echo "Model saved to: $OUTPUT_DIR/output/"
echo ""
echo "To test your model, use the TTS Python API:"
echo ""
echo "from TTS.api import TTS"
echo "model_path = \"$OUTPUT_DIR/output/best_model.pth\""
echo "config_path = \"$OUTPUT_DIR/output/config.json\""
echo "tts = TTS(model_path=model_path, config_path=config_path)"
echo "tts.tts_to_file(\"नमस्ते, कैसे हैं आप?\", speaker_wav=\"$OUTPUT_DIR/wavs/000001.wav\", language=\"hi\", file_path=\"output.wav\")"