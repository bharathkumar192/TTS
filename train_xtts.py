#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS fine-tuning training script for Hindi with multi-GPU support
"""

import os
import json
import argparse
import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.languages import get_language_id_from_name
from TTS.utils.audio import AudioProcessor
from TTS.utils.trainer import get_optimizer
from TTS.utils.distributed import init_distributed, reduce_tensor
import torch.distributed as dist
import torch.multiprocessing as mp


def download_xtts_model():
    """Download the XTTS v2 model if not already available"""
    manager = ModelManager()
    model_path = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    return model_path


def train(rank, world_size, args):
    """
    Main training function with distributed training support
    
    Args:
        rank: GPU rank for distributed training
        world_size: Total number of GPUs
        args: Command line arguments
    """
    # Initialize distributed training if using multiple GPUs
    if world_size > 1:
        init_distributed(rank, world_size, use_cuda=True)
        is_main_process = rank == 0
    else:
        is_main_process = True
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Create output directories
    output_path = Path(config_dict["output_path"])
    checkpoint_dir = output_path / "checkpoints"
    if is_main_process:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load XTTS config
    model_dir = Path(config_dict["model"]["base_model_path"])
    config_path = model_dir / "config.json"
    if not config_path.exists():
        config_path = Path(args.base_config_path)
    
    config = XttsConfig()
    config.load_json(config_path)
    
    # Update config with fine-tuning settings
    config.batch_size = config_dict["trainer"]["batch_size"]
    config.grad_clip = 5.0  # Add gradient clipping for stability
    config.lr = config_dict["trainer"]["lr"]
    config.optimizer = config_dict["trainer"]["optimizer"]
    config.scheduler = config_dict["trainer"]["scheduler"]
    config.epochs = config_dict["trainer"]["epochs"]
    config.language = "hi"  # Hindi language
    
    # Load data
    train_samples = load_tts_samples(
        Path(config_dict["data"]["path"]),
        Path(config_dict["data"]["audio_dir"]),
        eval_split=False
    )
    
    # If separate validation set is provided
    if args.val_metadata:
        eval_samples = load_tts_samples(
            Path(args.val_metadata),
            Path(config_dict["data"]["audio_dir"]),
            eval_split=True
        )
    else:
        # Use a subset of training data for validation
        np.random.shuffle(train_samples)
        val_size = int(len(train_samples) * 0.15)
        eval_samples = train_samples[:val_size]
        train_samples = train_samples[val_size:]
    
    if is_main_process:
        print(f"Number of training samples: {len(train_samples)}")
        print(f"Number of validation samples: {len(eval_samples)}")
    
    # Configure audio processor
    config.audio.sample_rate = config_dict["audio"]["sample_rate"]
    config.audio.win_length = config_dict["audio"]["win_length"]
    config.audio.hop_length = config_dict["audio"]["hop_length"]
    audio_processor = AudioProcessor.init_from_config(config)
    
    # Initialize or load model
    if args.checkpoint_path:
        # Load from checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model = Xtts.init_from_config(config)
        model.load_state_dict(checkpoint["model"])
        step = checkpoint.get("step", 0)
        epoch_start = checkpoint.get("epoch", 0)
        if is_main_process:
            print(f"Loaded checkpoint from {args.checkpoint_path}, starting from epoch {epoch_start}")
    else:
        # Initialize from base model
        model = Xtts.init_from_config(config)
        if model_dir.exists():
            try:
                model_checkpoint = torch.load(model_dir / "model.pth", map_location="cpu")
                model.load_state_dict(model_checkpoint, strict=False)
                if is_main_process:
                    print(f"Loaded base model from {model_dir}")
            except Exception as e:
                if is_main_process:
                    print(f"Error loading base model: {e}")
                raise e
        step = 0
        epoch_start = 0
    
    # Move model to device
    model = model.to(device)
    
    # Setup distributed model if using multiple GPUs
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            find_unused_parameters=True
        )
    
    # Initialize optimizer
    optimizer = get_optimizer(config.optimizer, model.parameters(), config.lr, weight_decay=1e-6)
    
    # Initialize scheduler
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Early stopping parameters
    early_stop_patience = config_dict["trainer"].get("early_stop_patience", 3)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epoch_start, config.epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle data for distributed training
        if world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_samples, num_replicas=world_size, rank=rank
            )
            train_loader = torch.utils.data.DataLoader(
                train_samples,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_samples,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        
        # Progress bar for main process only
        if is_main_process:
            train_progress_bar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{config.epochs}",
                position=0
            )
        
        # Accumulate gradients over specified steps
        grad_accum_steps = config_dict["trainer"].get("grad_accum_steps", 1)
        
        for batch_idx, batch in enumerate(train_loader):
            # Process batch (implement based on XTTS model requirements)
            text = batch["text"]
            wav = batch["wav"].to(device)
            wav_lens = batch["wav_length"].to(device)
            
            # Get Hindi language ID
            language_id = get_language_id_from_name("hi")
            language_id = torch.tensor([language_id] * len(text)).to(device)
            
            # Forward pass
            outputs = model(
                text,
                wav,
                wav_lens,
                language_id,
                None,  # No speaker embedding for fine-tuning
            )
            
            # Calculate loss
            loss = outputs["loss"] / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(train_loader) - 1:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
            
            # Reduce loss across GPUs
            if world_size > 1:
                reduced_loss = reduce_tensor(loss.data, world_size)
            else:
                reduced_loss = loss.data
            
            epoch_loss += reduced_loss.item() * grad_accum_steps
            
            # Update progress bar
            if is_main_process:
                train_progress_bar.update(1)
                train_progress_bar.set_postfix({"loss": reduced_loss.item() * grad_accum_steps})
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        
        if is_main_process:
            train_progress_bar.close()
            print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {epoch_loss:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        if world_size > 1:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_samples, num_replicas=world_size, rank=rank
            )
            eval_loader = torch.utils.data.DataLoader(
                eval_samples,
                batch_size=config.batch_size,
                sampler=eval_sampler,
                num_workers=2,
                pin_memory=True
            )
        else:
            eval_loader = torch.utils.data.DataLoader(
                eval_samples,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        if is_main_process:
            eval_progress_bar = tqdm(
                total=len(eval_loader),
                desc=f"Validating Epoch {epoch+1}",
                position=0
            )
        
        with torch.no_grad():
            for batch in eval_loader:
                text = batch["text"]
                wav = batch["wav"].to(device)
                wav_lens = batch["wav_length"].to(device)
                
                language_id = get_language_id_from_name("hi")
                language_id = torch.tensor([language_id] * len(text)).to(device)
                
                outputs = model(
                    text,
                    wav,
                    wav_lens,
                    language_id,
                    None,  # No speaker embedding for fine-tuning
                )
                
                # Calculate loss
                loss = outputs["loss"]
                
                # Reduce loss across GPUs
                if world_size > 1:
                    reduced_loss = reduce_tensor(loss.data, world_size)
                else:
                    reduced_loss = loss.data
                
                val_loss += reduced_loss.item()
                
                if is_main_process:
                    eval_progress_bar.update(1)
                    eval_progress_bar.set_postfix({"loss": reduced_loss.item()})
        
        # Calculate average validation loss
        val_loss /= len(eval_loader)
        
        if is_main_process:
            eval_progress_bar.close()
            print(f"Epoch {epoch+1}/{config.epochs} - Validation Loss: {val_loss:.6f}")
        
        # Apply scheduler
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if is_main_process:
            checkpoint = {
                "model": model.state_dict() if world_size == 1 else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch + 1,
                "config": config.to_dict(),
                "audio_processor": audio_processor.to_dict(),
            }
            
            # Save latest checkpoint
            torch.save(
                checkpoint,
                checkpoint_dir / f"checkpoint_latest.pth"
            )
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    checkpoint,
                    checkpoint_dir / f"checkpoint_best.pth"
                )
                print(f"Saved best model with validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Save epoch checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(
                    checkpoint,
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                )
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            if is_main_process:
                print(f"Early stopping after {patience_counter} epochs without improvement")
            break
    
    # Final evaluation or generation if needed
    if is_main_process:
        print("Training complete!")
        # Optional: generate samples from the final model
        generate_samples(
            model, 
            eval_samples[:5], 
            output_path / "samples", 
            device,
            audio_processor
        )


def generate_samples(model, samples, output_dir, device, audio_processor):
    """
    Generate audio samples from the trained model
    
    Args:
        model: Trained XTTS model
        samples: List of samples to generate audio for
        output_dir: Output directory for generated audio
        device: Device to run generation on
        audio_processor: Audio processor for audio processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            text = sample["text"]
            reference_wav = sample["wav"].unsqueeze(0).to(device)
            
            # Get Hindi language ID
            language_id = get_language_id_from_name("hi")
            language_id = torch.tensor([language_id]).to(device)
            
            # Generate audio
            outputs = model.inference(
                text,
                reference_wav,
                language_id,
                None,  # No speaker embedding for inference
            )
            
            # Save audio
            audio_processor.save_wav(
                outputs["wav"],
                os.path.join(output_dir, f"sample_{idx}.wav")
            )


def evaluate_metrics(model_path, test_metadata, audio_dir, output_dir, device="cuda"):
    """
    Evaluate the model using WER and speaker similarity metrics
    
    Args:
        model_path: Path to the trained model
        test_metadata: Path to test metadata CSV
        audio_dir: Path to test audio directory
        output_dir: Output directory for evaluation results
        device: Device to run evaluation on
    """
    import whisper
    from resemblyzer import VoiceEncoder
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = Xtts.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()
    
    # Load ASR model for WER
    asr_model = whisper.load_model("medium")
    
    # Load speaker encoder for similarity
    speaker_encoder = VoiceEncoder()
    
    # Load test samples
    test_samples = load_tts_samples(
        Path(test_metadata),
        Path(audio_dir),
        eval_split=True
    )
    
    results = {
        "sample_id": [],
        "reference_text": [],
        "transcribed_text": [],
        "wer": [],
        "speaker_similarity": []
    }
    
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
            text = sample["text"]
            reference_wav_path = sample["wav_path"]
            reference_wav = sample["wav"].unsqueeze(0).to(device)
            
            # Get Hindi language ID
            language_id = get_language_id_from_name("hi")
            language_id = torch.tensor([language_id]).to(device)
            
            # Generate audio
            outputs = model.inference(
                text,
                reference_wav,
                language_id,
                None,  # No speaker embedding for inference
            )
            
            # Save generated audio
            generated_wav_path = os.path.join(output_dir, f"gen_{idx}.wav")
            with open(generated_wav_path, "wb") as f:
                f.write(outputs["wav"])
            
            # Transcribe generated audio with Whisper
            transcription = asr_model.transcribe(generated_wav_path, language="hi")
            transcribed_text = transcription["text"]
            
            # Calculate WER
            # (Simplified WER calculation - might want to use a proper NLP library for this)
            ref_words = text.split()
            asr_words = transcribed_text.split()
            
            # Compute Levenshtein distance
            from Levenshtein import distance
            
            wer = distance(ref_words, asr_words) / len(ref_words)
            
            # Calculate speaker similarity
            ref_embedding = speaker_encoder.embed_utterance(audio_processor.load_wav(reference_wav_path))
            gen_embedding = speaker_encoder.embed_utterance(audio_processor.load_wav(generated_wav_path))
            
            similarity = np.inner(ref_embedding, gen_embedding)
            
            # Store results
            results["sample_id"].append(idx)
            results["reference_text"].append(text)
            results["transcribed_text"].append(transcribed_text)
            results["wer"].append(wer)
            results["speaker_similarity"].append(similarity)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Print summary
    avg_wer = results_df["wer"].mean()
    avg_similarity = results_df["speaker_similarity"].mean()
    
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average Speaker Similarity: {avg_similarity:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(results_df["wer"], bins=20)
    plt.title(f"Word Error Rate (Avg: {avg_wer:.4f})")
    plt.xlabel("WER")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.hist(results_df["speaker_similarity"], bins=20)
    plt.title(f"Speaker Similarity (Avg: {avg_similarity:.4f})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
    
    return avg_wer, avg_similarity


def main():
    parser = argparse.ArgumentParser(description="Train XTTS model for Hindi")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--base_config_path", type=str, help="Path to base XTTS config JSON file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to resume training")
    parser.add_argument("--val_metadata", type=str, help="Path to validation metadata CSV file")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    parser.add_argument("--test_metadata", type=str, help="Path to test metadata CSV file for evaluation")
    args = parser.parse_args()
    
    # Download XTTS model if not specified
    if not args.base_config_path and not args.checkpoint_path:
        try:
            model_path = download_xtts_model()
            args.base_config_path = os.path.join(model_path, "config.json")
            print(f"Downloaded XTTS model: {model_path}")
        except Exception as e:
            print(f"Error downloading XTTS model: {e}")
            print("Please specify --base_config_path or --checkpoint_path")
            return
    
    # Multi-GPU training
    if args.num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train, args=(args.num_gpus, args), nprocs=args.num_gpus)
    else:
        # Single GPU training
        train(0, 1, args)
    
    # Evaluation
    if args.evaluate and args.test_metadata:
        # Load config to get output path
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        output_path = Path(config_dict["output_path"])
        checkpoint_path = output_path / "checkpoints" / "checkpoint_best.pth"
        
        if os.path.exists(checkpoint_path):
            print(f"Evaluating model from {checkpoint_path}")
            evaluate_metrics(
                checkpoint_path,
                args.test_metadata,
                config_dict["data"]["audio_dir"],
                output_path / "evaluation",
            )
        else:
            print(f"Checkpoint not found at {checkpoint_path}, skipping evaluation")


if __name__ == "__main__":
    main()