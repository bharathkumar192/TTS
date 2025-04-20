# Simplified Guide: Hindi Voice Fine-tuning with Coqui XTTS

This guide provides a straightforward approach to fine-tune Coqui XTTS v2 for Hindi using the built-in tools. Instead of custom scripts, we'll use the official `tts finetune` command provided by Coqui TTS.

## 1. Installation and Setup

First, install Coqui TTS with all the necessary dependencies:

```bash
pip install TTS==0.22.0
pip install torch torchaudio
```

Verify your installation:

```bash
tts --list_models
```

This should display available models, including XTTS v2.

## 2. Data Preparation

Coqui TTS expects your data in a specific format. Let's prepare it:

### 2.1 Convert Your JSON Data to Metadata CSV

Create a simple script to convert your JSON format to Coqui's required format:

```python
# json_to_metadata.py
import json
import os

# Load your JSON data
with open('final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create metadata entries
metadata = []
for entry in data:
    sentence_id = entry["sentence_id"]
    hindi_sentence = entry["hindi_sentence"]
    
    # Check if audio file exists
    audio_file = f"final_audio_24khz/{sentence_id}.wav"
    if os.path.exists(audio_file):
        # Format: file_id|text
        metadata.append(f"{sentence_id}|{hindi_sentence}")

# Write metadata file
with open('metadata.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(metadata))

print(f"Created metadata.csv with {len(metadata)} entries")
```

Run this script to create your metadata file:

```bash
python json_to_metadata.py
```

### 2.2 Organize Audio Files

Ensure your audio files are organized correctly:

```bash
# Create 'wavs' directory if it doesn't exist
mkdir -p dataset/wavs

# Copy and/or resample your audio files to 24kHz if needed
# For simplicity, we'll link existing files
for wav in final_audio_24khz/*.wav; do
    filename=$(basename "$wav")
    ln -s "$(realpath $wav)" "dataset/wavs/$filename"
done
```

### 2.3 Create Dataset Config

Create a small configuration file for your dataset:

```bash
echo '{
  "meta_file_train": "metadata.csv",
  "meta_file_val": null,
  "path": "dataset/wavs",
  "language": "hi"
}' > dataset_config.json
```

## 3. Fine-tuning Configuration

Create a configuration file for fine-tuning:

```bash
echo '{
  "run_name": "xtts_hindi_finetune",
  "output_path": "xtts_hindi_output",
  "batch_size": 8,
  "eval_batch_size": 8,
  "num_loader_workers": 4,
  "num_eval_loader_workers": 2,
  "epochs": 10,
  "lr": 1e-5,
  "optimizer": "AdamW",
  "scheduler": "CosineAnnealingLR",
  "warmup_steps": 0,
  "grad_clip": 5.0,
  "early_stopping_patience": 3,
  "mixed_precision": true,
  "save_n_checkpoints": 3,
  "save_best_after": 1000,
  "print_step": 50,
  "language": "hi"
}' > config.json
```

## 4. Running Fine-tuning

Now you can fine-tune the model using Coqui's built-in command:

```bash
# For single GPU
tts finetune --config_path config.json \
  --model_path tts_models/multilingual/multi-dataset/xtts_v2 \
  --dataset_config_path dataset_config.json \
  --use_cuda true

# For multiple GPUs
tts finetune --config_path config.json \
  --model_path tts_models/multilingual/multi-dataset/xtts_v2 \
  --dataset_config_path dataset_config.json \
  --use_cuda true \
  --strategy ddp \
  --num_gpus 2
```

This will start the fine-tuning process using Coqui's built-in training pipeline. The model will be saved in the `output_path` specified in your config.

## 5. Monitoring Training

During training, you can monitor progress through:

1. Terminal output showing loss values and ETA
2. TensorBoard logs (if TensorBoard is installed):

```bash
pip install tensorboard
tensorboard --logdir xtts_hindi_output/tensorboard
```

## 6. Testing Your Fine-tuned Model

After training, test your model with:

```python
import torch
from TTS.api import TTS

# Initialize TTS with your fine-tuned model
model_path = "xtts_hindi_output/best_model.pth"
config_path = "xtts_hindi_output/config.json"
tts = TTS(model_path=model_path, config_path=config_path)

# Generate speech
reference_wav = "dataset/wavs/000001.wav"  # Choose a reference file
output_wav = "test_output.wav"

tts.tts_to_file(
    text="आपका स्वागत है", 
    speaker_wav=reference_wav,
    language="hi",
    file_path=output_wav
)
```

## 7. Evaluation

To evaluate your model, you can use built-in metrics or external tools:

### 7.1 WER (Word Error Rate)

Using Whisper for Hindi ASR:

```bash
pip install openai-whisper
```

```python
import whisper

# Load ASR model
model = whisper.load_model("medium")

# Transcribe generated audio
result = model.transcribe("test_output.wav", language="hi")
print(f"Transcription: {result['text']}")

# Compare with original text to calculate WER manually
# or use specialized WER calculation libraries
```

### 7.2 Speaker Similarity

```bash
pip install resemblyzer
```

```python
from resemblyzer import VoiceEncoder
import librosa

# Load voice encoder
encoder = VoiceEncoder()

# Calculate embeddings
original_wav, _ = librosa.load("dataset/wavs/000001.wav", sr=16000)
generated_wav, _ = librosa.load("test_output.wav", sr=16000)

original_embedding = encoder.embed_utterance(original_wav)
generated_embedding = encoder.embed_utterance(generated_wav)

# Calculate cosine similarity
import numpy as np
similarity = np.inner(original_embedding, generated_embedding)
print(f"Speaker similarity: {similarity:.4f}")
```

## 8. All-in-One Shell Script

Here's a complete script that performs all steps:

```bash
#!/bin/bash
# XTTS Hindi Fine-tuning Script

# 1. Install dependencies
pip install TTS==0.22.0 torch torchaudio openai-whisper resemblyzer

# 2. Create directories
mkdir -p dataset/wavs

# 3. Convert JSON to metadata.csv
python json_to_metadata.py

# 4. Link audio files to dataset directory
for wav in final_audio_24khz/*.wav; do
    filename=$(basename "$wav")
    ln -s "$(realpath $wav)" "dataset/wavs/$filename"
done

# 5. Create dataset config
echo '{
  "meta_file_train": "metadata.csv",
  "meta_file_val": null,
  "path": "dataset/wavs",
  "language": "hi"
}' > dataset_config.json

# 6. Create training config
echo '{
  "run_name": "xtts_hindi_finetune",
  "output_path": "xtts_hindi_output",
  "batch_size": 8,
  "eval_batch_size": 8,
  "num_loader_workers": 4,
  "num_eval_loader_workers": 2,
  "epochs": 10,
  "lr": 1e-5,
  "optimizer": "AdamW",
  "scheduler": "CosineAnnealingLR",
  "warmup_steps": 0,
  "grad_clip": 5.0,
  "early_stopping_patience": 3,
  "mixed_precision": true,
  "save_n_checkpoints": 3,
  "save_best_after": 1000,
  "print_step": 50,
  "language": "hi"
}' > config.json

# 7. Run fine-tuning
tts finetune --config_path config.json \
  --model_path tts_models/multilingual/multi-dataset/xtts_v2 \
  --dataset_config_path dataset_config.json \
  --use_cuda true

# 8. Test model
echo "Fine-tuning complete! Test your model with:"
echo "python test_model.py"
```

Create a test script:

```python
# test_model.py
import torch
from TTS.api import TTS

# Initialize TTS with your fine-tuned model
model_path = "xtts_hindi_output/best_model.pth"
config_path = "xtts_hindi_output/config.json"
tts = TTS(model_path=model_path, config_path=config_path)

# Generate speech
reference_wav = "dataset/wavs/000001.wav"  # Choose a reference file
output_wav = "test_output.wav"

tts.tts_to_file(
    text="आपका स्वागत है", 
    speaker_wav=reference_wav,
    language="hi",
    file_path=output_wav
)

print(f"Generated audio saved to {output_wav}")
```

## 9. Recommended Fine-tuning Parameters

For your 10-20 hours of Hindi data:

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Batch Size | 8 | Reduce to 4 if OOM errors |
| Learning Rate | 1e-5 | Good balance for fine-tuning |
| Epochs | 10 | Usually sufficient for 10+ hours |
| Early Stopping | 3 epochs | Prevents overfitting |
| Gradient Clipping | 5.0 | Stabilizes training |

## 10. Troubleshooting

1. **Out of Memory Errors**:
   - Reduce batch size
   - Set `mixed_precision: true` in config
   - Increase gradient accumulation steps

2. **Poor Quality Output**:
   - Try longer training (increase epochs)
   - Use a longer reference audio clip (6-10 seconds)
   - Check audio quality and transcriptions

3. **Training Stuck or Errors**:
   - Check CUDA version compatibility
   - Ensure data format is correct
   - Try different optimizer settings

