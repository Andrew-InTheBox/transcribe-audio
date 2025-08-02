#!/usr/bin/env python3
"""
Audio processing pipeline for OpenAI Whisper transcription.
Orchestrates preprocessing and transcription of WAV files.
"""

import os
import subprocess
import sys
import glob
import shutil
from pathlib import Path

class AudioProcessingPipeline:
    def __init__(self, audio_dir="./audio-files", output_dir="./output", models_dir="./models"):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.processed_dir = self.audio_dir / "processed"
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
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
        """Run Whisper transcription using Docker."""
        print(f"Running Whisper transcription on: {processed_file}")
        
        # Create output directory for this file
        file_output_dir = self.output_dir / processed_file.stem
        file_output_dir.mkdir(exist_ok=True)
        
        # Docker command for Whisper transcription
        cmd = [
            "docker", "run", "--gpus", "all", "-it",
            "-v", f"{self.models_dir.absolute()}:/root/.cache/whisper",
            "-v", f"{self.processed_dir.absolute()}:/app/input",
            "-v", f"{file_output_dir.absolute()}:/app/output",
            "openai-whisper",
            "whisper", f"/app/input/{processed_file.name}",
            "--device", "cuda",
            "--model", "large-v3",
            "--language", "en",
            "--output_dir", "/app/output",
            "--output_format", "txt"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Transcription completed: {file_output_dir}")
            return file_output_dir
        except subprocess.CalledProcessError as e:
            print(f"Error running Whisper transcription: {e}")
            print(f"Error output: {e.stderr}")
            return None
    
    def process_all_files(self):
        """Process all new WAV files in the pipeline."""
        new_files = self.find_new_wav_files()
        
        if not new_files:
            print("No new WAV files found to process.")
            return
        
        print(f"Found {len(new_files)} new WAV files to process:")
        for file in new_files:
            print(f"  - {file.name}")
        
        processed_count = 0
        failed_count = 0
        
        for wav_file in new_files:
            print(f"\n{'='*60}")
            print(f"Processing: {wav_file.name}")
            print(f"{'='*60}")
            
            # Step 1: Normalize audio
            normalized_file = self.normalize_audio(wav_file)
            if not normalized_file:
                print(f"Failed to normalize {wav_file.name}, skipping...")
                failed_count += 1
                continue
            
            # Step 2: Run Whisper transcription
            output_dir = self.run_whisper_transcription(normalized_file)
            if not output_dir:
                print(f"Failed to transcribe {wav_file.name}")
                failed_count += 1
                continue
            
            processed_count += 1
            print(f"Successfully processed {wav_file.name}")
        
        print(f"\n{'='*60}")
        print(f"Pipeline completed:")
        print(f"  - Successfully processed: {processed_count} files")
        print(f"  - Failed: {failed_count} files")
        print(f"  - Output directory: {self.output_dir.absolute()}")
        print(f"{'='*60}")
    
    def list_status(self):
        """Show status of files in the pipeline."""
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
            
            normalized_status = "✓" if normalized_file.exists() else "✗"
            transcribed_status = "✓" if output_dir.exists() and any(output_dir.glob("*.txt")) else "✗"
            
            print(f"{wav_file.name:<30} {normalized_status:<12} {transcribed_status:<12}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        pipeline = AudioProcessingPipeline()
        pipeline.list_status()
    elif len(sys.argv) > 1 and sys.argv[1] == "help":
        print("Audio Processing Pipeline")
        print("Usage:")
        print("  python process_audio_pipeline.py        - Process all new WAV files")
        print("  python process_audio_pipeline.py status - Show status of all files")
        print("  python process_audio_pipeline.py help   - Show this help")
        print("")
        print("Pipeline process:")
        print("1. Finds new .wav files in ./audio-files/")
        print("2. Normalizes audio using normalize_simple.py")
        print("3. Runs Whisper transcription with large-v3 model")
        print("4. Outputs results to ./output/[filename]/")
    else:
        # Run the full pipeline
        pipeline = AudioProcessingPipeline()
        pipeline.process_all_files()

if __name__ == "__main__":
    main()