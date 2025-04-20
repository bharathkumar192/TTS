#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for XTTS Hindi fine-tuned model
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from resemblyzer import VoiceEncoder
import whisper
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.languages import get_language_id_from_name
from TTS.tts.datasets import load_tts_samples


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate using Levenshtein distance
    
    Args:
        reference: Reference text
        hypothesis: Transcribed text
    
    Returns:
        WER: Word Error Rate
    """
    from Levenshtein import distance
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 1.0
    
    return distance(ref_words, hyp_words) / len(ref_words)


def generate_samples(model, test_samples, audio_processor, output_dir, device="cuda"):
    """
    Generate audio samples from test data
    
    Args:
        model: XTTS model
        test_samples: Test samples
        audio_processor: Audio processor
        output_dir: Output directory
        device: Device to run generation on
    
    Returns:
        List of generated sample paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    generated_paths = []
    
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_samples, desc="Generating samples")):
            text = sample["text"]
            reference_wav = AudioProcessor.load_wav(sample["wav_path"], 
                                               audio_processor.sample_rate)
            reference_wav = torch.FloatTensor(reference_wav).unsqueeze(0).to(device)
            
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
            generated_path = os.path.join(output_dir, f"gen_{idx:04d}.wav")
            sf.write(generated_path, outputs["wav"], audio_processor.sample_rate)
            
            generated_paths.append({
                "index": idx,
                "text": text,
                "reference_path": sample["wav_path"],
                "generated_path": generated_path
            })
    
    return generated_paths


def evaluate_wer(generated_samples, output_dir, whisper_model_size="medium"):
    """
    Evaluate Word Error Rate using Whisper ASR
    
    Args:
        generated_samples: List of generated samples
        output_dir: Output directory for results
        whisper_model_size: Size of Whisper model to use
    
    Returns:
        DataFrame with WER results
    """
    print(f"Loading Whisper model ({whisper_model_size})...")
    asr_model = whisper.load_model(whisper_model_size)
    
    results = []
    
    for sample in tqdm(generated_samples, desc="Calculating WER"):
        reference_text = sample["text"]
        
        # Transcribe generated audio with Whisper
        transcription = asr_model.transcribe(
            sample["generated_path"], 
            language="hi",
            fp16=torch.cuda.is_available()
        )
        transcribed_text = transcription["text"]
        
        # Calculate WER
        wer = calculate_wer(reference_text, transcribed_text)
        
        results.append({
            "index": sample["index"],
            "reference_text": reference_text,
            "transcribed_text": transcribed_text,
            "wer": wer,
            "reference_path": sample["reference_path"],
            "generated_path": sample["generated_path"]
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_csv_path = os.path.join(output_dir, "wer_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    # Calculate average WER
    avg_wer = results_df["wer"].mean()
    print(f"Average WER: {avg_wer:.4f}")
    
    # Save plot
    plt.figure(figsize=(10, 6))
    plt.hist(results_df["wer"], bins=20)
    plt.title(f"Word Error Rate Distribution (Avg: {avg_wer:.4f})")
    plt.xlabel("WER")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "wer_distribution.png"))
    
    return results_df


def evaluate_speaker_similarity(generated_samples, output_dir):
    """
    Evaluate speaker similarity using Resemblyzer
    
    Args:
        generated_samples: List of generated samples
        output_dir: Output directory for results
    
    Returns:
        DataFrame with speaker similarity results
    """
    print("Loading speaker encoder...")
    speaker_encoder = VoiceEncoder()
    
    results = []
    
    for sample in tqdm(generated_samples, desc="Calculating speaker similarity"):
        try:
            # Calculate speaker embeddings
            reference_wav = speaker_encoder.preprocess_wav(sample["reference_path"])
            generated_wav = speaker_encoder.preprocess_wav(sample["generated_path"])
            
            reference_embedding = speaker_encoder.embed_utterance(reference_wav)
            generated_embedding = speaker_encoder.embed_utterance(generated_wav)
            
            # Calculate cosine similarity
            similarity = np.inner(reference_embedding, generated_embedding)
            
            results.append({
                "index": sample["index"],
                "reference_path": sample["reference_path"],
                "generated_path": sample["generated_path"],
                "speaker_similarity": similarity
            })
        except Exception as e:
            print(f"Error processing sample {sample['index']}: {e}")
            results.append({
                "index": sample["index"],
                "reference_path": sample["reference_path"],
                "generated_path": sample["generated_path"],
                "speaker_similarity": 0.0
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_csv_path = os.path.join(output_dir, "speaker_similarity_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    # Calculate average similarity
    avg_similarity = results_df["speaker_similarity"].mean()
    print(f"Average Speaker Similarity: {avg_similarity:.4f}")
    
    # Save plot
    plt.figure(figsize=(10, 6))
    plt.hist(results_df["speaker_similarity"], bins=20)
    plt.title(f"Speaker Similarity Distribution (Avg: {avg_similarity:.4f})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "speaker_similarity_distribution.png"))
    
    return results_df


def measure_real_time_factor(model, test_samples, audio_processor, device="cuda", num_samples=10):
    """
    Measure Real-Time Factor (RTF) of the model
    
    Args:
        model: XTTS model
        test_samples: Test samples
        audio_processor: Audio processor
        device: Device to run inference on
        num_samples: Number of samples to test
    
    Returns:
        Average RTF
    """
    import time
    
    model.eval()
    rtf_values = []
    
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_samples[:num_samples], desc="Measuring RTF")):
            text = sample["text"]
            reference_wav = AudioProcessor.load_wav(sample["wav_path"], 
                                              audio_processor.sample_rate)
            reference_wav = torch.FloatTensor(reference_wav).unsqueeze(0).to(device)
            
            # Get Hindi language ID
            language_id = get_language_id_from_name("hi")
            language_id = torch.tensor([language_id]).to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model.inference(
                text,
                reference_wav,
                language_id,
                None,  # No speaker embedding for inference
            )
            inference_time = time.time() - start_time
            
            # Calculate audio duration
            audio_duration = len(outputs["wav"]) / audio_processor.sample_rate
            
            # Calculate RTF
            rtf = inference_time / audio_duration
            rtf_values.append(rtf)
    
    avg_rtf = np.mean(rtf_values)
    print(f"Average Real-Time Factor: {avg_rtf:.4f}")
    
    return avg_rtf


def combine_evaluation_results(wer_results, similarity_results, output_dir):
    """
    Combine WER and speaker similarity results
    
    Args:
        wer_results: WER results DataFrame
        similarity_results: Speaker similarity results DataFrame
        output_dir: Output directory for combined results
    
    Returns:
        Combined results DataFrame
    """
    # Merge DataFrames on index
    combined_results = pd.merge(
        wer_results,
        similarity_results[["index", "speaker_similarity"]],
        on="index"
    )
    
    # Save combined results
    combined_csv_path = os.path.join(output_dir, "combined_evaluation_results.csv")
    combined_results.to_csv(combined_csv_path, index=False)
    
    # Create scatter plot of WER vs Speaker Similarity
    plt.figure(figsize=(10, 8))
    plt.scatter(combined_results["speaker_similarity"], combined_results["wer"], alpha=0.6)
    plt.title("WER vs Speaker Similarity")
    plt.xlabel("Speaker Similarity (cosine)")
    plt.ylabel("Word Error Rate")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "wer_vs_similarity.png"))
    
    # Calculate average metrics
    avg_wer = combined_results["wer"].mean()
    avg_similarity = combined_results["speaker_similarity"].mean()
    
    # Create summary bar chart
    plt.figure(figsize=(8, 6))
    metrics = ["WER", "Speaker Similarity"]
    values = [avg_wer, avg_similarity]
    plt.bar(metrics, values, color=["#ff7675", "#74b9ff"])
    plt.title("Evaluation Metrics Summary")
    plt.ylim(0, 1.0)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f"{v:.4f}", ha="center")
    
    plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
    
    # Create summary file
    with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
        f.write(f"Average WER: {avg_wer:.4f}\n")
        f.write(f"Average Speaker Similarity: {avg_similarity:.4f}\n")
        
        # Calculate percentiles
        wer_p25 = combined_results["wer"].quantile(0.25)
        wer_p50 = combined_results["wer"].quantile(0.50)
        wer_p75 = combined_results["wer"].quantile(0.75)
        
        sim_p25 = combined_results["speaker_similarity"].quantile(0.25)
        sim_p50 = combined_results["speaker_similarity"].quantile(0.50)
        sim_p75 = combined_results["speaker_similarity"].quantile(0.75)
        
        f.write(f"\nWER Percentiles:\n")
        f.write(f"25th: {wer_p25:.4f}\n")
        f.write(f"50th: {wer_p50:.4f}\n")
        f.write(f"75th: {wer_p75:.4f}\n")
        
        f.write(f"\nSpeaker Similarity Percentiles:\n")
        f.write(f"25th: {sim_p25:.4f}\n")
        f.write(f"50th: {sim_p50:.4f}\n")
        f.write(f"75th: {sim_p75:.4f}\n")
        
        # Print model quality assessment
        f.write("\nModel Quality Assessment:\n")
        if avg_wer <= 0.05 and avg_similarity >= 0.85:
            quality = "Excellent"
        elif avg_wer <= 0.08 and avg_similarity >= 0.75:
            quality = "Good"
        elif avg_wer <= 0.12 and avg_similarity >= 0.65:
            quality = "Acceptable"
        else:
            quality = "Needs Improvement"
        
        f.write(f"Overall Quality: {quality}\n")
    
    return combined_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate XTTS Hindi fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_metadata", type=str, required=True, help="Path to test metadata CSV")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to test audio directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation results")
    parser.add_argument("--whisper_model", type=str, default="medium", help="Whisper model size for ASR")
    parser.add_argument("--rtf_samples", type=int, default=10, help="Number of samples for RTF calculation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = Xtts.load_from_checkpoint(args.model_path)
    model = model.to(args.device)
    model.eval()
    
    # Extract audio processor from model
    config = model.config
    audio_processor = AudioProcessor.init_from_config(config)
    
    # Load test samples
    print(f"Loading test samples from {args.test_metadata}")
    test_samples = load_tts_samples(
        Path(args.test_metadata),
        Path(args.audio_dir),
        eval_split=True
    )
    
    print(f"Found {len(test_samples)} test samples")
    
    # Generate samples
    print("Generating samples...")
    generated_samples = generate_samples(
        model,
        test_samples,
        audio_processor,
        os.path.join(args.output_dir, "generated_samples"),
        device=args.device
    )
    
    # Evaluate WER
    print("Evaluating Word Error Rate (WER)...")
    wer_results = evaluate_wer(
        generated_samples,
        args.output_dir,
        whisper_model_size=args.whisper_model
    )
    
    # Evaluate speaker similarity
    print("Evaluating speaker similarity...")
    similarity_results = evaluate_speaker_similarity(
        generated_samples,
        args.output_dir
    )
    
    # Measure RTF
    print("Measuring Real-Time Factor (RTF)...")
    rtf = measure_real_time_factor(
        model,
        test_samples,
        audio_processor,
        device=args.device,
        num_samples=args.rtf_samples
    )
    
    # Combine results
    print("Combining evaluation results...")
    combined_results = combine_evaluation_results(
        wer_results,
        similarity_results,
        args.output_dir
    )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    
    # Write RTF to summary file
    with open(os.path.join(args.output_dir, "evaluation_summary.txt"), "a") as f:
        f.write(f"\nReal-Time Factor: {rtf:.4f}\n")
        f.write(f"(RTF < 1.0 means faster than real-time)\n")
    
    # Print final summary
    print("\nEvaluation Summary:")
    print(f"Average WER: {wer_results['wer'].mean():.4f}")
    print(f"Average Speaker Similarity: {similarity_results['speaker_similarity'].mean():.4f}")
    print(f"Real-Time Factor: {rtf:.4f}")


if __name__ == "__main__":
    main()