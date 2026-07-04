#!/usr/bin/env python3
"""
Audio processing pipeline for OpenAI Whisper transcription.
Orchestrates preprocessing and transcription of WAV files.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

class AudioProcessingPipeline:
    def __init__(self, audio_dir="./audio-files", output_dir="./output", model_name="large-v3"):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.audio_dir / "processed"
        self.model = None
        self.model_name = model_name

        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

    def find_new_video_files(self):
        """Find video files that haven't had their audio extracted yet."""
        video_files = [
            f for f in self.audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]

        new_files = []
        for video_file in video_files:
            extracted_file = self.audio_dir / f"{video_file.stem}.wav"
            if not extracted_file.exists():
                new_files.append(video_file)

        return new_files

    def extract_audio_from_video(self, video_file):
        """Use ffmpeg to extract a WAV audio track from a video file."""
        print(f"Extracting audio from video: {video_file.name}")

        output_file = self.audio_dir / f"{video_file.stem}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Audio extraction completed: {output_file}")
            return output_file
        except FileNotFoundError:
            print("Error: ffmpeg not found. Install ffmpeg and ensure it's on your PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {video_file.name}: {e}")
            print(f"Error output: {e.stderr}")
            return None
    
    def _load_model(self):
        """Load Whisper model with error handling."""
        if self.model is not None:
            return  # Already loaded
            
        try:
            print(f"Loading Whisper model '{self.model_name}'...")
            import whisper
            # Model selection is configurable via CLI flag or constructor parameter
            self.model = whisper.load_model(self.model_name)
            print("Model loaded successfully")
        except ImportError:
            print("Error: whisper not installed. Run: pip install -U openai-whisper")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Try using a smaller model like 'turbo' or 'base' if you have memory issues")
            sys.exit(1)
        
    def find_new_wav_files(self):
        """Find new WAV files that haven't been processed yet."""
        wav_files = list(self.audio_dir.glob("*.wav"))
        
        # Filter out already processed files
        new_files = []
        for wav_file in wav_files:
            processed_file = self.processed_dir / f"{wav_file.stem}_enhanced_norm.wav"
            if not processed_file.exists():
                new_files.append(wav_file)
        
        return new_files

    def find_pending_transcriptions(self):
        """Find normalized files that still need transcription."""
        if not self.processed_dir.exists():
            return []

        pending = []
        for processed_file in self.processed_dir.glob("*_enhanced_norm.wav"):
            output_dir = self.output_dir / processed_file.stem
            if not output_dir.exists() or not any(output_dir.glob("*.txt")):
                pending.append(processed_file)

        return pending
    
    def normalize_audio(self, input_file):
        """Run the normalization preprocessing script."""
        print(f"Normalizing audio: {input_file}")
        
        output_file = self.processed_dir / f"{input_file.stem}_enhanced_norm.wav"
        
        # Run normalize_simple.py script
        cmd = [
            sys.executable, "normalize_simple.py", 
            str(input_file), 
            str(output_file),
            "-6.0"  # target dB level
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Normalization completed: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error normalizing {input_file}: {e}")
            print(f"Error output: {e.stderr}")
            return None
    
    def run_whisper_transcription(self, processed_file):
        """Run Whisper transcription directly."""
        # Load model only when needed for transcription
        self._load_model()
        
        print(f"Running Whisper transcription on: {processed_file}")
        
        # Create output directory for this file
        file_output_dir = self.output_dir / processed_file.stem
        file_output_dir.mkdir(exist_ok=True)
        
        try:
            # Transcribe audio using the model
            result = self.model.transcribe(
                str(processed_file),
                language="en",
                verbose=True
            )
            
            # Save transcription
            output_file = file_output_dir / f"{processed_file.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            print(f"Transcription completed: {output_file}")
            return file_output_dir
            
        except Exception as e:
            print(f"Error running Whisper transcription: {e}")
            return None
    
    def process_all_files(self, normalize_only=False):
        """Process all new video/WAV files in the pipeline."""
        new_videos = self.find_new_video_files()
        extraction_failures = 0

        if new_videos:
            print(f"Found {len(new_videos)} new video file(s) to extract audio from:")
            for video_file in new_videos:
                print(f"  - {video_file.name}")

            for video_file in new_videos:
                print(f"\n{'='*60}")
                print(f"Extracting audio: {video_file.name}")
                print(f"{'='*60}")

                if not self.extract_audio_from_video(video_file):
                    print(f"Failed to extract audio from {video_file.name}, skipping...")
                    extraction_failures += 1

        new_files = self.find_new_wav_files()
        pending_transcriptions = self.find_pending_transcriptions()

        if not new_files and not pending_transcriptions:
            print("No new WAV files found to process.")
            return

        if new_files:
            print(f"Found {len(new_files)} new WAV files to process:")
            for file in new_files:
                print(f"  - {file.name}")
        else:
            print("No new WAV files require normalization.")

        if pending_transcriptions:
            print(f"Found {len(pending_transcriptions)} normalized files awaiting transcription.")

        if normalize_only:
            pending_transcriptions = []

        normalized_success = 0
        normalization_failures = 0
        transcription_success = 0
        transcription_failures = 0
        processed_this_run = set()

        for wav_file in new_files:
            print(f"\n{'='*60}")
            print(f"Processing: {wav_file.name}")
            print(f"{'='*60}")

            normalized_file = self.normalize_audio(wav_file)
            if not normalized_file:
                print(f"Failed to normalize {wav_file.name}, skipping transcription...")
                normalization_failures += 1
                continue

            normalized_success += 1
            processed_this_run.add(normalized_file)

            if normalize_only:
                continue

            output_dir = self.run_whisper_transcription(normalized_file)
            if not output_dir:
                print(f"Failed to transcribe {wav_file.name}")
                transcription_failures += 1
                continue

            transcription_success += 1
            print(f"Successfully processed {wav_file.name}")

        for processed_file in pending_transcriptions:
            if processed_file in processed_this_run:
                continue  # Already handled above

            print(f"\n{'='*60}")
            print(f"Transcribing existing normalized file: {processed_file.name}")
            print(f"{'='*60}")

            output_dir = self.run_whisper_transcription(processed_file)
            if not output_dir:
                print(f"Failed to transcribe {processed_file.stem}")
                transcription_failures += 1
                continue

            transcription_success += 1
            print(f"Successfully transcribed {processed_file.name}")

        print(f"\n{'='*60}")
        print("Pipeline completed:")
        if new_videos:
            print(f"  - Videos with audio extracted: {len(new_videos) - extraction_failures} files")
        print(f"  - Newly normalized: {normalized_success} files")
        if not normalize_only:
            print(f"  - Transcribed: {transcription_success} files")
        if extraction_failures:
            print(f"  - Extraction failures: {extraction_failures}")
        if normalization_failures:
            print(f"  - Normalization failures: {normalization_failures}")
        if transcription_failures:
            print(f"  - Transcription failures: {transcription_failures}")
        print(f"  - Output directory: {self.output_dir.absolute()}")
        print(f"{'='*60}")
    
    def list_status(self):
        """Show status of files in the pipeline."""
        video_files = [
            f for f in self.audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]

        if video_files:
            print("Video files status:")
            print(f"{'File':<30} {'Extracted':<12}")
            print("-" * 42)
            for video_file in video_files:
                extracted_file = self.audio_dir / f"{video_file.stem}.wav"
                extracted_status = "yes" if extracted_file.exists() else "no"
                print(f"{video_file.name:<30} {extracted_status:<12}")
            print()

        wav_files = list(self.audio_dir.glob("*.wav"))

        if not wav_files:
            print("No WAV files found in audio-files directory.")
            return

        print(f"Audio files status:")
        print(f"{'File':<30} {'Normalized':<12} {'Transcribed':<12}")
        print("-" * 54)

        for wav_file in wav_files:
            normalized_file = self.processed_dir / f"{wav_file.stem}_enhanced_norm.wav"
            output_dir = self.output_dir / f"{wav_file.stem}_enhanced_norm"
            
            normalized_status = "yes" if normalized_file.exists() else "no"
            transcribed_status = "yes" if output_dir.exists() and any(output_dir.glob("*.txt")) else "no"
            
            print(f"{wav_file.name:<30} {normalized_status:<12} {transcribed_status:<12}")

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Audio Processing Pipeline for OpenAI Whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Optional command: 'status' to list files or 'help' for pipeline details",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="large-v3",
        help="Whisper model name (e.g. 'medium.en', 'medium', 'large-v3', 'turbo')",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Only extract/normalize audio; skip Whisper transcription",
    )
    parsed = parser.parse_args(args)
    return parser, parsed


def main():
    parser, parsed_args = parse_args(sys.argv[1:])
    command = parsed_args.command
    model_name = parsed_args.model

    if command == "help":
        print("Audio Processing Pipeline")
        print("Usage:")
        print("  python process_audio_pipeline.py [--model MODEL]")
        print("  python process_audio_pipeline.py --normalize-only")
        print("  python process_audio_pipeline.py status [--model MODEL]")
        print("  python process_audio_pipeline.py help")
        print("")
        print("Pipeline process:")
        print("1. Finds new video files (.mp4, .mov, .mkv, .avi, .webm, .m4v) in ./audio-files/")
        print("   and extracts their audio to .wav using ffmpeg")
        print("2. Finds new .wav files in ./audio-files/")
        print("3. Normalizes audio using normalize_simple.py")
        print("4. Runs Whisper transcription with the selected model")
        print("5. Outputs results to ./output/[filename]/")
        print("")
        print("Examples:")
        print("  python process_audio_pipeline.py --model medium.en")
        print("  python process_audio_pipeline.py --model large-v3")
        return

    if command and command not in {"status", None}:
        parser.error(f"Unknown command: {command}")

    pipeline = AudioProcessingPipeline(model_name=model_name)

    if command == "status":
        pipeline.list_status()
    else:
        pipeline.process_all_files(normalize_only=parsed_args.normalize_only)

if __name__ == "__main__":
    main()
