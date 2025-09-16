# Audio Transcription Pipeline with OpenAI Whisper

An automated pipeline that processes WAV audio files through enhancement and transcription using OpenAI Whisper. Optimizes field recordings for speech recognition and outputs text transcriptions.

## What It Does

1. **Audio Enhancement**: Normalizes levels, filters noise, compresses dynamic range  
2. **Transcription**: Uses OpenAI Whisper to convert speech to text  
3. **Organization**: Saves results in organized folders

## Setup

### Prerequisites
- Python 3.8-3.11  
- NVIDIA GPU recommended (but not required)  
- FFmpeg

The pipeline runs as a regular Python process. Docker is only used if you want to invoke the Whisper CLI inside a container rather than through the locally installed Python package.

### Installation (local Python)

1. **Create virtual environment**:
```bash
python -m venv whisper-env
source whisper-env/bin/activate  # Linux/Mac
# whisper-env\Scripts\activate   # Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg**:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# MacOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

4. **Create directories**:
```bash
mkdir audio-files output
```

## Usage

1. **Add WAV files** to `./audio-files/` folder

2. **Run pipeline**:
```bash
python process_audio_pipeline.py
```

3. **Check status**:
```bash
python process_audio_pipeline.py status
```

## Output

- Enhanced audio: `./audio-files/processed/`  
- Transcriptions: `./output/[filename]/`

## Docker Usage (optional)

The included `Dockerfile` builds an image that only contains the Whisper CLI. You can use it if you prefer to run transcription inside a container while still running the Python preprocessing locally.

```bash
# Build the image
docker build -t whisper-cli .

# Run Whisper against a processed file
docker run --rm \
  -v "$PWD/audio-files/processed":/audio \
  -v "$PWD/output":/out \
  whisper-cli --model large-v3 /audio/myfile_enhanced_norm.wav --output_dir /out
```

When using the pipeline end-to-end (`python process_audio_pipeline.py`), Whisper will be loaded from your local Python environment instead of Docker.

## Models

The pipeline defaults to `medium.en`, but you can pick any Whisper model at runtime:

- `tiny`, `base`, `small` (and their `.en` variants) – lightweight, fastest  
- `medium`, `medium.en` – balanced accuracy (~5 GB VRAM)  
- `large-v3`, `large-v3-turbo` – highest accuracy (~10 GB VRAM)  
- `turbo` – very fast multilingual model

Override with `--model`, e.g. `python process_audio_pipeline.py --model large-v3`.

## Directory Structure
```
./
├── audio-files/          # Put WAV files here
│   └── processed/        # Enhanced audio (auto-created)
├── output/               # Transcription results
├── process_audio_pipeline.py
└── normalize_simple.py
```
