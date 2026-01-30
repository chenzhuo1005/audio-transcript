# Audio Transcript

A Python project for transcribing audio files using OpenAI's Whisper model. This tool automatically downloads and manages Whisper models, and outputs transcriptions to text files.

## Features

- 🎙️ Transcribe audio files to text using Whisper models
- 📦 Automatic model download and caching
- ⚙️ Configurable model selection via JSON config
- 📁 Organized model and output directory management
- 🚀 Simple command-line interface

## Requirements

- Python 3.8+
- Poetry (for dependency management)

## Installation

1. Install Poetry if you haven't already:
   ```bash
   pip install poetry
   ```

2. Install project dependencies:
   ```bash
   poetry install
   ```

## Configuration

Edit `config.json` to configure the Whisper model and directories:

```json
{
  "model": "large-v3",
  "model_dir": "models",
  "output_dir": "outputs",
  "input_dir": "audio_files",
  "device": "auto"
}
```

### Device (GPU/CPU)

- **`auto`** (default) - Use GPU (CUDA) if available, otherwise CPU
- **`cuda`** - Force GPU (requires NVIDIA GPU + CUDA + PyTorch with CUDA)
- **`cpu`** - Force CPU

Whisper does not use GPU by default; this project automatically loads the model to CUDA when available. Install PyTorch with CUDA support if you have an NVIDIA GPU (e.g. `pip install torch` often picks the right build; for explicit CUDA see [PyTorch install](https://pytorch.org/get-started/locally/)).

### Available Models

- `tiny` - Fastest, least accurate (~39M parameters)
- `base` - Balanced speed and accuracy (~74M parameters) - **Default**
- `small` - Better accuracy (~244M parameters)
- `medium` - High accuracy (~769M parameters)
- `large` - Best accuracy (~1550M parameters)

## Usage

### Default Behavior (No Arguments)

If no arguments are provided, the script will automatically transcribe all audio files in the `audio_files/` directory:

```bash
poetry run python transcribe.py
```

Place your audio files in the `audio_files/` folder and run the script without arguments for batch processing.

### Single File Transcription

```bash
poetry run python transcribe.py <audio_file_path>
```

The output will be automatically saved to `outputs/<audio_filename>.txt`.

### Specify Output Path

```bash
poetry run python transcribe.py <audio_file_path> <output_file_path>
```

### Batch Directory Processing

Transcribe all audio files in a specified directory:

```bash
poetry run python transcribe.py <input_directory>
```

All audio files in the directory will be transcribed and saved to the `outputs/` directory with corresponding `.txt` files.

### Examples

```bash
# Default: transcribe all files in audio_files/ directory
poetry run python transcribe.py

# Transcribe single file (outputs to outputs/audio.txt)
poetry run python transcribe.py audio.mp3

# Transcribe with custom output path
poetry run python transcribe.py audio.mp3 my_transcript.txt

# Transcribe different audio formats
poetry run python transcribe.py recording.wav
poetry run python transcribe.py podcast.m4a

# Batch transcribe all audio files in a custom directory
poetry run python transcribe.py ./my_audio_folder/
```

## How It Works

1. **Model Loading**: The script checks if the specified Whisper model exists in the `models/` directory
2. **Auto Download**: If the model doesn't exist, it automatically downloads it
3. **Transcription**: The audio file is processed using the loaded model
4. **Output**: The transcription result is saved to a text file

## Project Structure

```
audio-transcript/
├── pyproject.toml      # Poetry configuration and dependencies
├── config.json         # Configuration file
├── transcribe.py       # Main transcription script
├── audio_files/        # Default input directory for audio files
├── models/             # Whisper models storage (gitignored)
└── outputs/            # Transcription output files (gitignored)
```

## Supported Audio Formats

Whisper supports various audio formats including:
- MP3
- WAV
- M4A
- FLAC
- OGG
- And more formats supported by ffmpeg

## Notes

- Models are cached in the `models/` directory after first download
- The `models/` and `outputs/` directories are gitignored
- Larger models provide better accuracy but require more processing time and memory

## License

See LICENSE file for details.
