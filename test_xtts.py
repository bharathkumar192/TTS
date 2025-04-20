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

def evaluate_model(model_dir, reference_wav=None, test_sentences=None):
    """
    Evaluate the fine-tuned XTTS model
    
    Args:
        model_dir: Directory containing the fine-tuned model
        reference_wav: Path to reference wav file (if None, will choose randomly)
        test_sentences: List of test sentences (if None, will use default)
    """
    try:
        import whisper
        from resemblyzer import VoiceEncoder
        import numpy as np
        import librosa
        import time
    except ImportError:
        print("Error: Required packages for evaluation not found.")
        print("Please install them with: pip install openai-whisper resemblyzer numpy librosa")
        return
    
    # Find the model files (same as test_model)
    checkpoint_dir = os.path.join(model_dir, "output")
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Could not find model directory at {checkpoint_dir}")
        return
    
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(model_path):
        # Try to find checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
        if not checkpoint_files:
            print(f"Error: No model checkpoints found in {checkpoint_dir}")
            return
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
    
    # Default test sentences if not provided
    if test_sentences is None:
        test_sentences = [
            "नमस्ते, मेरा नाम क्या है?",
            "आप कैसे हैं? मुझे आशा है कि आप अच्छे हैं।",
            "यह एक परीक्षण वाक्य है जिसे हमारे फाइन-ट्यून्ड मॉडल द्वारा उत्पन्न किया जाएगा।"
        ]
    
    # Initialize TTS with fine-tuned model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_path=model_path, config_path=config_path).to(device)
    
    # Load ASR model for WER
    print("Loading Whisper ASR model...")
    asr_model = whisper.load_model("medium")
    
    # Load voice encoder for similarity
    print("Loading voice encoder for similarity...")
    encoder = VoiceEncoder()
    
    # Load reference embedding
    reference_audio, _ = librosa.load(reference_wav, sr=16000)
    reference_embedding = encoder.embed_utterance(reference_audio)
    
    # Create output directory
    eval_dir = os.path.join(model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    results = []
    
    print("\nEvaluating model...")
    for i, sentence in enumerate(test_sentences):
        print(f"\nGenerating speech for sentence {i+1}/{len(test_sentences)}")
        output_file = os.path.join(eval_dir, f"eval_{i+1}.wav")
        
        # Measure generation time
        start_time = time.time()
        tts.tts_to_file(
            text=sentence,
            speaker_wav=reference_wav,
            language="hi",
            file_path=output_file
        )
        generation_time = time.time() - start_time
        
        # Calculate audio duration
        audio, sr = librosa.load(output_file, sr=None)
        audio_duration = librosa.get_duration(y=audio, sr=sr)
        
        # Calculate Real-Time Factor
        rtf = generation_time / audio_duration
        
        # Transcribe using Whisper
        print("  Transcribing with Whisper...")
        transcription = asr_model.transcribe(output_file, language="hi")
        transcribed_text = transcription["text"]
        
        # Calculate speaker similarity
        print("  Calculating speaker similarity...")
        generated_audio, _ = librosa.load(output_file, sr=16000)
        generated_embedding = encoder.embed_utterance(generated_audio)
        speaker_similarity = np.inner(reference_embedding, generated_embedding)
        
        # Store results
        results.append({
            "sentence": sentence,
            "transcription": transcribed_text,
            "rtf": rtf,
            "speaker_similarity": speaker_similarity
        })
        
        print(f"  Generated: {output_file}")
        print(f"  Original: {sentence}")
        print(f"  Transcription: {transcribed_text}")
        print(f"  RTF: {rtf:.4f}")
        print(f"  Speaker Similarity: {speaker_similarity:.4f}")
    
    # Calculate averages
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_similarity = sum(r["speaker_similarity"] for r in results) / len(results)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Average RTF: {avg_rtf:.4f} (lower is better, <1.0 means faster than real-time)")
    print(f"Average Speaker Similarity: {avg_similarity:.4f} (higher is better, >0.75 is good)")
    
    # Save results to file
    with open(os.path.join(eval_dir, "evaluation_results.txt"), "w") as f:
        f.write("=== Evaluation Results ===\n\n")
        f.write(f"Average RTF: {avg_rtf:.4f}\n")
        f.write(f"Average Speaker Similarity: {avg_similarity:.4f}\n\n")
        
        for i, r in enumerate(results):
            f.write(f"--- Sentence {i+1} ---\n")
            f.write(f"Original: {r['sentence']}\n")
            f.write(f"Transcription: {r['transcription']}\n")
            f.write(f"RTF: {r['rtf']:.4f}\n")
            f.write(f"Speaker Similarity: {r['speaker_similarity']:.4f}\n\n")
    
    print(f"\nDetailed results saved to {os.path.join(eval_dir, 'evaluation_results.txt')}")
    return results

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <model_dir> [reference_wav] [output_file] [--evaluate]")
        sys.exit(1)
    
    # Directory containing the fine-tuned model
    model_dir = sys.argv[1]
    
    # Check if evaluation is requested
    if "--evaluate" in sys.argv:
        # Remove --evaluate from arguments
        sys.argv.remove("--evaluate")
        
        # Reference wav file (optional)
        reference_wav = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Run evaluation
        evaluate_model(model_dir, reference_wav)
    else:
        # Reference wav file (optional)
        reference_wav = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Output file (optional)
        output_file = sys.argv[3] if len(sys.argv) > 3 else "test_output.wav"
        
        # Run test
        test_model(model_dir, reference_wav, output_file)