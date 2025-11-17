#!/usr/bin/env python3
"""
Extract and synthesize audio from HDF5 files in the timer_audio dataset.
"""

import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy.io import wavfile
import glob

def extract_audio_from_hdf5(hdf5_path):
    """
    Extract audio data from a single HDF5 file and save as WAV in the same directory.
    
    Args:
        hdf5_path: Path to the HDF5 file
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'audio' not in f:
                print(f"Warning: No audio data found in {hdf5_path}")
                return False
            
            # Get audio data and metadata
            audio_data = f['audio'][:]  # Shape: (num_timesteps, max_samples)
            sampling_rate = f.attrs.get('audio_sampling_rate', 48000)
            channels = f.attrs.get('audio_channels', 1)
            
            print(f"Processing {os.path.basename(hdf5_path)}:")
            print(f"  Audio shape: {audio_data.shape}")
            print(f"  Sampling rate: {sampling_rate}Hz")
            print(f"  Channels: {channels}")
            
            # Extract and concatenate audio data
            concatenated_audio = []
            total_samples = 0
            
            for t in range(audio_data.shape[0]):
                # Get actual length (stored in last element)
                actual_length = audio_data[t, -1]
                if actual_length > 0:
                    # Extract valid audio data
                    timestep_audio = audio_data[t, :actual_length]
                    concatenated_audio.append(timestep_audio)
                    total_samples += actual_length
                else:
                    # No audio data for this timestep
                    print(f"  Warning: Timestep {t} has no audio data")
            
            if not concatenated_audio:
                print(f"  Error: No valid audio data found in {hdf5_path}")
                return False
            
            # Concatenate all audio data
            full_audio = np.concatenate(concatenated_audio)
            print(f"  Total audio samples: {len(full_audio)}")
            print(f"  Audio duration: {len(full_audio) / sampling_rate:.2f} seconds")
            
            # Convert to appropriate format for WAV
            if full_audio.dtype != np.int16:
                # Convert to int16 if needed
                if full_audio.dtype == np.float32:
                    # Assume it's normalized to [-1, 1]
                    full_audio = (full_audio * 32767).astype(np.int16)
                else:
                    full_audio = full_audio.astype(np.int16)
            
            # Save as WAV file in the same directory as the HDF5 file
            output_dir = os.path.dirname(hdf5_path)
            output_filename = os.path.basename(hdf5_path).replace('.hdf5', '.wav')
            output_path = os.path.join(output_dir, output_filename)
            
            wavfile.write(output_path, sampling_rate, full_audio)
            print(f"  Saved audio to: {output_path}")
            
            return True
            
    except Exception as e:
        print(f"Error processing {hdf5_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract audio from HDF5 files')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing HDF5 files (WAV files will be saved in the same directory)'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Find all HDF5 files
    hdf5_pattern = os.path.join(args.input_dir, '*.hdf5')
    hdf5_files = glob.glob(hdf5_pattern)
    
    if not hdf5_files:
        print(f"No HDF5 files found in {args.input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    print(f"Input directory: {args.input_dir}")
    print(f"WAV files will be saved in the same directory as HDF5 files")
    print("-" * 60)
    
    # Process each HDF5 file
    successful = 0
    failed = 0
    
    for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 files"):
        if extract_audio_from_hdf5(hdf5_file):
            successful += 1
        else:
            failed += 1
        print("-" * 60)
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(hdf5_files)}")

if __name__ == "__main__":
    main()
