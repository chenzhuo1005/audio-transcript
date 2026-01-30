#!/usr/bin/env python3
"""
Audio transcription tool
Transcribe audio files to text using Whisper models
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List
import torch
import whisper


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid configuration file format: {e}")
        sys.exit(1)


def get_model_path(model_name: str, model_dir: str) -> Path:
    """Get model path"""
    return Path(model_dir) / model_name


def get_device(device_preference: str = "auto") -> str:
    """Get compute device (cuda or cpu).
    
    Args:
        device_preference: "cuda", "cpu", or "auto" (use GPU if available)
    
    Returns:
        Device string for model.to(device)
    """
    if device_preference == "cpu":
        return "cpu"
    if device_preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("Warning: CUDA requested but not available, falling back to CPU")
        return "cpu"
    # auto: use GPU if available
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_or_download_model(model_name: str, model_dir: str, device: str = "auto"):
    """Load or download Whisper model
    
    Args:
        model_name: Whisper model name (tiny, base, small, medium, large)
        model_dir: Model storage directory
        device: "cuda", "cpu", or "auto" (use GPU if available)
    
    Returns:
        Whisper model object
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(exist_ok=True)
    
    # Whisper handles model caching automatically, but we set custom cache directory
    os.environ["WHISPER_CACHE_DIR"] = str(model_dir_path.absolute())
    
    compute_device = get_device(device)
    print(f"Loading model: {model_name}")
    print(f"Model directory: {model_dir_path.absolute()}")
    print(f"Device: {compute_device}" + (f" ({torch.cuda.get_device_name(0)})" if compute_device == "cuda" else ""))
    
    try:
        # Load model (will auto-download if not exists)
        model = whisper.load_model(model_name, download_root=str(model_dir_path))
        model = model.to(compute_device)
        print(f"Model loaded successfully: {model_name}")
        return model
    except Exception as e:
        print(f"Error: Failed to load model {model_name}: {e}")
        sys.exit(1)


def get_audio_files(directory: str) -> List[Path]:
    """Get all audio files from a directory
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of audio file paths
    """
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', 
                        '.wma', '.mp4', '.m4v', '.webm', '.opus', '.amr'}
    
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    
    audio_files = []
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)
    
    return sorted(audio_files)


def transcribe_audio(audio_path: str, model, output_path: str = None, show_result: bool = True):
    """Transcribe audio file
    
    Args:
        audio_path: Path to audio file
        model: Whisper model object
        output_path: Output text file path (optional)
        show_result: Whether to print transcription result (default: True)
    
    Returns:
        Transcribed text
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Transcribing: {audio_path}")
    print(f"{'='*60}")
    
    try:
        # Load audio once to get duration (Whisper uses 16kHz mono)
        audio = whisper.load_audio(audio_path)
        duration_sec = len(audio) / 16000
        duration_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"
        print(f"Audio duration: {duration_str}")
        print("Transcribing (progress bar with ETA below)...")
        
        start_time = time.perf_counter()
        # verbose=False enables tqdm progress bar with frames/s and ETA
        result = model.transcribe(audio, verbose=False)
        elapsed = time.perf_counter() - start_time
        
        text = result["text"]
        
        print("\nTranscription completed!")
        print(f"Total time: {elapsed:.1f}s")
        if show_result:
            print(f"Transcription result:\n{text}\n")
        
        # Save to file
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Result saved to: {output_path}")
        
        return text
    except Exception as e:
        print(f"Error: Transcription failed: {e}")
        return None


def main():
    """Main function"""
    # Load configuration
    config = load_config()
    default_input_dir = config.get("input_dir", "audio_files")
    model_name = config.get("model", "base")
    model_dir = config.get("model_dir", "models")
    output_dir = config.get("output_dir", "outputs")
    device = config.get("device", "auto")
    
    if len(sys.argv) < 2:
        # No arguments provided, use default input directory
        input_path = default_input_dir
        output_path = None
        print(f"No input specified, using default directory: {input_path}")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load or download model (only once for batch processing)
    model = load_or_download_model(model_name, model_dir, device)
    
    # Check if input is a directory
    input_path_obj = Path(input_path)
    if input_path_obj.is_dir():
        # Batch processing: transcribe all audio files in directory
        audio_files = get_audio_files(input_path)
        
        if not audio_files:
            print(f"Error: No audio files found in directory: {input_path}")
            sys.exit(1)
        
        print(f"\nFound {len(audio_files)} audio file(s) in directory: {input_path}")
        print("Starting batch transcription...\n")
        
        successful = 0
        failed = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            # Auto-generate output path
            output_filename = audio_file.stem + ".txt"
            file_output_path = os.path.join(output_dir, output_filename)
            
            # Transcribe (don't show full result for batch processing)
            result = transcribe_audio(str(audio_file), model, file_output_path, show_result=False)
            
            if result:
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Batch transcription completed!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
    elif input_path_obj.is_file():
        # Single file processing
        # Auto-generate output path if not specified
        if output_path is None:
            audio_file = Path(input_path)
            output_filename = audio_file.stem + ".txt"
            output_path = os.path.join(output_dir, output_filename)
        
        # Execute transcription
        transcribe_audio(input_path, model, output_path)
    else:
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
