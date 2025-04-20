# XTTS Hindi Fine-tuning on Google Colab
# Run this entire script in a Colab notebook

import os
import json
import random
import shutil
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
# from google.colab import drive, files

# Mount Google Drive to access your data
# drive.mount('/content/drive')

# Install required packages
# !pip install TTS torch accelerate

# 1. Set up paths and parameters
# Change these paths to match your data location
DATA_ROOT = ""  # This will be where we put the processed data
JSON_PATH = "./final.json"  # Path to your JSON file
AUDIO_DIR = "./final_audio_24khz"  # Path to your audio directory
OUTPUT_DIR = "./xtts_hindi_finetune"
SAMPLE_RATE = 24000
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
VALIDATION_RATIO = 0.15

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "wavs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "output"), exist_ok=True)

# 2. Prepare the dataset
print("Loading JSON data...")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Found {len(data)} entries in JSON file")

# Function to check if audio file exists and copy it
def process_audio_file(entry):
    sentence_id = entry["sentence_id"]
    hindi_sentence = entry["hindi_sentence"]
    
    src_path = os.path.join(AUDIO_DIR, f"{sentence_id}.wav")
    dst_path = os.path.join(OUTPUT_DIR, "wavs", f"{sentence_id}.wav")
    
    if os.path.exists(src_path):
        # Check if audio needs resampling
        try:
            info = torchaudio.info(src_path)
            if info.sample_rate != SAMPLE_RATE:
                # Resample audio
                waveform, sr = torchaudio.load(src_path)
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
                torchaudio.save(dst_path, waveform, SAMPLE_RATE)
            else:
                # Just copy the file
                shutil.copy(src_path, dst_path)
            return True, sentence_id, hindi_sentence
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
            return False, None, None
    else:
        print(f"Warning: Audio file {src_path} not found")
        return False, None, None

# Process all audio files and create metadata entries
print("Processing audio files...")
metadata_entries = []
for entry in tqdm(data):
    success, sentence_id, hindi_sentence = process_audio_file(entry)
    if success:
        metadata_entries.append(f"{sentence_id}|{hindi_sentence}")

print(f"Successfully processed {len(metadata_entries)} audio files")

# Split into train and validation sets
random.shuffle(metadata_entries)
val_size = int(len(metadata_entries) * VALIDATION_RATIO)
train_entries = metadata_entries[val_size:]
val_entries = metadata_entries[:val_size]

# Write metadata files
train_csv_path = os.path.join(OUTPUT_DIR, "metadata_train.csv")
val_csv_path = os.path.join(OUTPUT_DIR, "metadata_val.csv")

with open(train_csv_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_entries))

with open(val_csv_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_entries))

print(f"Created training metadata with {len(train_entries)} entries at {train_csv_path}")
print(f"Created validation metadata with {len(val_entries)} entries at {val_csv_path}")

# 3. Create dataset config
dataset_config = {
    "meta_file_train": "metadata_train.csv",
    "meta_file_val": "metadata_val.csv",
    "path": "wavs",
    "language": "hi"
}

dataset_config_path = os.path.join(OUTPUT_DIR, "dataset_config.json")
with open(dataset_config_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_config, f, indent=2)

# 4. Setup training with TTS directly 
print("Setting up fine-tuning...")

# Download base XTTS model
from TTS.utils.manage import ModelManager
model_path = ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
print(f"Downloaded base model to {model_path}")

# Import TTS modules
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets.dataset import load_tts_samples
from TTS.tts.utils.languages import get_language_id_from_name
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.trainer import Trainer, TrainerArgs

# Load the base model config
config = XttsConfig()
config_path = os.path.join(model_path, "config.json")
config.load_json(config_path)

# Update config with fine-tuning parameters
config.audio.sample_rate = SAMPLE_RATE
config.batch_size = BATCH_SIZE
config.eval_batch_size = BATCH_SIZE
config.mixed_precision = True
config.run_eval = True
config.epochs = NUM_EPOCHS
config.lr = LEARNING_RATE
config.optimizer = "AdamW"
config.optimizer_params = {"weight_decay": 1e-6}
config.scheduler = "CosineAnnealingLR"
config.scheduler_params = {"T_max": NUM_EPOCHS}
config.grad_clip = 5.0
config.early_stopping_patience = 3
config.save_n_checkpoints = 3
config.target_loss = "loss"
config.print_step = 50
config.mixed_precision = True
config.language = "hi"

# Save the config for training
config_output_path = os.path.join(OUTPUT_DIR, "config.json")
config.save_json(config_output_path)
print(f"Saved config to {config_output_path}")

# Setup dataset
print("Loading training samples...")
train_samples = load_tts_samples(
    Path(train_csv_path),
    Path(os.path.join(OUTPUT_DIR, "wavs")),
    eval_split=False
)

print("Loading validation samples...")
eval_samples = load_tts_samples(
    Path(val_csv_path),
    Path(os.path.join(OUTPUT_DIR, "wavs")),
    eval_split=True
)

# Initialize audio processor
audio_processor = AudioProcessor.init_from_config(config)

# Initialize the model
print("Initializing model...")
model = Xtts.init_from_config(config)

# Load the base model weights
model_checkpoint = torch.load(os.path.join(model_path, "model.pth"), map_location="cpu")
model.load_state_dict(model_checkpoint, strict=False)
print("Loaded base model weights")

# Setup trainer arguments
trainer_args = TrainerArgs(
    restore_path=None,
    skip_train_epoch=False,
    start_epoch=0,
    epochs=NUM_EPOCHS,
    use_grad_scaler=True,
    grad_accum_steps=1,
    batch_size=BATCH_SIZE,
    mixed_precision=True,
    eval_batch_size=BATCH_SIZE,
    num_eval_loader_workers=4,
    output_path=os.path.join(OUTPUT_DIR, "output"),
    logger_uri=None,
    print_step=50,
    plot_step=100,
    model_param_stats=False,
    dashboard_logger="tensorboard",
    lr=LEARNING_RATE,
    optimizer="AdamW",
    optimizer_params={"betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 1e-6},
    scheduler="CosineAnnealingLR",
    scheduler_params={"T_max": NUM_EPOCHS, "eta_min": 1e-7},
    use_accelerate=True,  # Use accelerate library
    accelerate_config=None,
    reinit_optimizer=True,
    precision="fp16"  # Use mixed precision for faster training
)

# Setup trainer
trainer = Trainer(
    trainer_args,
    config,
    output_path=os.path.join(OUTPUT_DIR, "output"),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={
        "audio_processor": audio_processor
    }
)

# 5. Start fine-tuning
print("Starting fine-tuning...")
try:
    trainer.fit()
    print("Fine-tuning completed successfully!")
except Exception as e:
    print(f"An error occurred during training: {e}")

# 6. Function to test the model
def test_model():
    print("Testing fine-tuned model...")
    
    # Find the best model
    best_model_path = os.path.join(OUTPUT_DIR, "output", "best_model.pth")
    if not os.path.exists(best_model_path):
        checkpoint_files = [f for f in os.listdir(os.path.join(OUTPUT_DIR, "output")) if f.startswith("checkpoint_") and f.endswith(".pth")]
        if checkpoint_files:
            best_model_path = os.path.join(OUTPUT_DIR, "output", sorted(checkpoint_files)[-1])
            print(f"Best model not found, using latest checkpoint: {best_model_path}")
        else:
            print("No model checkpoints found!")
            return
    
    # Get a reference audio file
    reference_wavs = os.listdir(os.path.join(OUTPUT_DIR, "wavs"))
    if not reference_wavs:
        print("No reference audio files found!")
        return
    
    reference_wav = os.path.join(OUTPUT_DIR, "wavs", reference_wavs[0])
    
    # Initialize TTS with the fine-tuned model
    from TTS.api import TTS
    tts = TTS(model_path=best_model_path, config_path=config_output_path)
    
    # Test sentences
    test_sentences = [
        "नमस्ते, कैसे हैं आप?",
        "हिंदी भाषा में यह मॉडल बहुत अच्छा काम करता है।"
    ]
    
    for i, sentence in enumerate(test_sentences):
        output_path = f"/content/test_output_{i+1}.wav"
        
        print(f"Generating speech for: {sentence}")
        tts.tts_to_file(
            text=sentence,
            speaker_wav=reference_wav,
            language="hi",
            file_path=output_path
        )
        
        # Play the audio
        from IPython.display import Audio, display
        print(f"Playing generated audio:")
        display(Audio(output_path))
        
        # Download option
        files.download(output_path)

# Test the model after training
test_model()

print("All training and testing complete!")