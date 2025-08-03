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

### Installation

1. **Create virtual environment**:
```bash
python -m venv whisper-env
source whisper-env/bin/activate  # Linux/Mac
# whisper-env\Scripts\activate   # Windows
```

2. **Install dependencies**:
```bash
pip install openai-whisper soundfile scipy numpy
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

## Models

The pipeline uses `large-v3` by default. To use a faster model, edit line 24 in the script:

- `turbo` - Fast, good accuracy (~6GB VRAM)  
- `medium` - Balanced (~5GB VRAM)  
- `small` - Lightweight (~2GB VRAM)

## Directory Structure
```
./
├── audio-files/          # Put WAV files here
│   └── processed/        # Enhanced audio (auto-created)
├── output/               # Transcription results
├── process_audio_pipeline.py
└── normalize_simple.py
```
