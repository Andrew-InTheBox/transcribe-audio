#!/usr/bin/env python3
"""
Enhanced audio normalization script with compression for speech transcription.
"""

import soundfile as sf
import numpy as np
import sys
import os
from scipy import signal

def apply_high_pass_filter(audio, samplerate, cutoff=80):
    """Apply high-pass filter to remove low-frequency noise."""
    nyquist = samplerate / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, audio, axis=0)

def soft_compress(audio, threshold=0.7, ratio=3.0):
    """Apply soft compression to reduce dynamic range."""
    # Convert to dB, apply compression, convert back
    audio_abs = np.abs(audio)
    mask = audio_abs > threshold
    
    compressed = audio.copy()
    if np.any(mask):
        # Compress signals above threshold
        over_threshold = audio_abs[mask]
        compressed_level = threshold + (over_threshold - threshold) / ratio
        gain = compressed_level / over_threshold
        compressed[mask] = audio[mask] * gain
    
    return compressed

def calculate_speech_rms(input_file, info, noise_floor=0.005):
    """Calculate RMS level focusing on speech content, ignoring silence."""
    print("Analyzing audio content...")
    
    # Sample more sections for better coverage
    samples = []
    sample_count = min(50, int(info.duration))  # More samples for longer files
    
    for i in range(sample_count):
        start = i * info.frames // sample_count
        chunk, _ = sf.read(input_file, start=start, frames=22050)  # 0.5 sec each
        samples.append(chunk)
    
    all_samples = np.vstack(samples)
    
    # Focus on non-silent parts for RMS calculation
    non_silent = all_samples[np.abs(all_samples) > noise_floor]
    
    if len(non_silent) > len(all_samples) * 0.1:  # At least 10% should be non-silent
        current_rms = np.sqrt(np.mean(non_silent**2))
        print(f"Using speech-focused RMS from {len(non_silent)/len(all_samples)*100:.1f}% of samples")
    else:
        # Fallback to all samples if mostly silent
        current_rms = np.sqrt(np.mean(all_samples**2))
        print("Using full-signal RMS (mostly silent audio detected)")
    
    return current_rms

def normalize_audio_enhanced(input_file, output_file=None, target_db=-6.0, 
                           enable_compression=True, enable_hpf=True):
    """Enhanced normalization with compression and filtering for speech."""
    
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_enhanced_norm.wav"
    
    print(f"Processing: {input_file}")
    
    # Get file info
    info = sf.info(input_file)
    print(f"File: {info.channels} channels, {info.samplerate}Hz, {info.duration/60:.1f} min")
    
    # Calculate speech-focused RMS
    current_rms = calculate_speech_rms(input_file, info)
    current_rms_db = 20 * np.log10(current_rms) if current_rms > 0 else -100
    
    print(f"Current speech RMS level: {current_rms_db:.1f} dB")
    
    # Calculate gain to reach target RMS
    target_rms = 10 ** (target_db / 20)
    base_gain = target_rms / current_rms
    
    print(f"Target RMS: {target_db} dB")
    print(f"Base gain factor: {base_gain:.2f}")
    
    # Process file in chunks
    chunk_size = 44100 * 5  # 5 seconds for better processing
    
    with sf.SoundFile(output_file, 'w', samplerate=info.samplerate, 
                      channels=info.channels, subtype='PCM_16') as output:
        
        total_chunks = (info.frames + chunk_size - 1) // chunk_size
        
        for chunk_idx, start in enumerate(range(0, info.frames, chunk_size)):
            if chunk_idx % 20 == 0:  # Progress indicator
                print(f"Processing chunk {chunk_idx + 1}/{total_chunks}...")
            
            frames_to_read = min(chunk_size, info.frames - start)
            chunk, _ = sf.read(input_file, start=start, frames=frames_to_read)
            
            # Apply high-pass filter to remove low-frequency noise
            if enable_hpf:
                chunk = apply_high_pass_filter(chunk, info.samplerate)
            
            # Apply base gain
            chunk = chunk * base_gain
            
            # Apply soft compression to control dynamics
            if enable_compression:
                chunk = soft_compress(chunk, threshold=0.6, ratio=2.5)
            
            # Final safety limiter (soft clipping)
            chunk = np.tanh(chunk * 0.95) * 0.95
            
            output.write(chunk)
    
    # Verify result
    print("Verifying output...")
    verify_frames = min(44100 * 10, info.frames)  # Check first 10 seconds
    verify_data, _ = sf.read(output_file, frames=verify_frames)
    
    # Calculate final levels
    final_rms = np.sqrt(np.mean(verify_data**2))
    final_rms_db = 20 * np.log10(final_rms) if final_rms > 0 else -100
    final_peak = np.max(np.abs(verify_data))
    final_peak_db = 20 * np.log10(final_peak) if final_peak > 0 else -100
    
    print(f"\nResults:")
    print(f"Final RMS: {final_rms_db:.1f} dB (target: {target_db} dB)")
    print(f"Final peak: {final_peak_db:.1f} dB")
    print(f"Peak-to-RMS ratio: {final_peak_db - final_rms_db:.1f} dB")
    
    # File sizes
    input_size = os.path.getsize(input_file) / (1024**3)
    output_size = os.path.getsize(output_file) / (1024**3)
    print(f"File size: {input_size:.2f} GB -> {output_size:.2f} GB")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python normalize_enhanced.py <input_file> [output_file] [target_db]")
        print("  target_db: Target RMS level in dB (default: -6.0)")
        print("  Example: python normalize_enhanced.py speech.wav speech_loud.wav -3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    target_db = float(sys.argv[3]) if len(sys.argv) > 3 else -6.0
    
    # Validate target level
    if target_db > 0:
        print("Warning: Target level above 0 dB may cause distortion")
    elif target_db < -20:
        print("Warning: Target level very low, may not improve transcription")
    
    try:
        result = normalize_audio_enhanced(input_file, output_file, target_db)
        print(f"\nSuccess! Enhanced file: {result}")
        print("\nOptimizations applied:")
        print("- Speech-focused RMS calculation")
        print("- High-pass filtering (80Hz cutoff)")
        print("- Dynamic range compression")
        print("- Soft limiting for distortion prevention")
        print("- Higher target level (-6 dB default)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)