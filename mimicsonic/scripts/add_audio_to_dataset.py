#!/usr/bin/env python3
"""
Script to add raw audio data to existing HDF5 datasets.
This script creates new dataset files with raw audio data added to the obs groups.
"""

import h5py
import numpy as np
import os
import shutil
from pathlib import Path
import argparse


def generate_raw_audio(seq_length, sample_rate=16000, audio_length=2.0):
    """
    Generate random raw audio data for a given sequence length.
    
    Args:
        seq_length (int): Length of the sequence
        sample_rate (int): Audio sample rate
        audio_length (float): Audio length in seconds
    
    Returns:
        np.ndarray: Random raw audio data with shape (seq_length, audio_samples)
    """
    # Calculate number of audio samples
    audio_samples = int(sample_rate * audio_length)
    
    # Generate random raw audio data
    # Using normal distribution with mean=0, std=0.1, which is typical for normalized audio
    raw_audio = np.random.normal(0, 0.1, (seq_length, audio_samples)).astype(np.float32)
    
    # Add some structure to make it more realistic
    # Add sinusoidal components to simulate speech-like patterns
    t = np.linspace(0, audio_length, audio_samples)
    
    for i in range(seq_length):
        # Add multiple frequency components
        audio = np.zeros(audio_samples)
        
        # Add fundamental frequency (speech-like)
        audio += 0.3 * np.sin(2 * np.pi * 200 * t)  # 200 Hz fundamental
        
        # Add harmonics
        audio += 0.2 * np.sin(2 * np.pi * 400 * t)  # 400 Hz harmonic
        audio += 0.1 * np.sin(2 * np.pi * 600 * t)  # 600 Hz harmonic
        
        # Add some noise
        audio += np.random.normal(0, 0.05, audio_samples)
        
        # Apply envelope to make it more speech-like
        envelope = np.exp(-t * 0.5) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio *= envelope
        
        raw_audio[i] = audio
    
    return raw_audio


def copy_hdf5_with_audio(input_path, output_path, audio_data_generator=None, audio_length=2.0, sample_rate=16000):
    """
    Copy HDF5 file and add audio data to obs groups.
    
    Args:
        input_path (str): Path to input HDF5 file
        output_path (str): Path to output HDF5 file
        audio_data_generator (callable): Function to generate audio data
        audio_length (float): Audio length in seconds
        sample_rate (int): Audio sample rate in Hz
    """
    print(f"Processing: {input_path} -> {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(input_path, 'r') as src_file:
        with h5py.File(output_path, 'w') as dst_file:
            # Copy all groups and datasets recursively
            def copy_item(name, obj):
                if isinstance(obj, h5py.Group):
                    # Create group
                    if name not in dst_file:
                        dst_file.create_group(name)
                    
                    # Copy group attributes (this is crucial for num_samples!)
                    for attr_key, attr_value in obj.attrs.items():
                        dst_file[name].attrs[attr_key] = attr_value
                    
                    # If this is an obs group, add audio data
                    if name.endswith('/obs'):
                        print(f"  Adding audio data to: {name}")
                        
                        # Copy all existing obs data
                        for key in obj.keys():
                            src_dataset = obj[key]
                            dst_file[name].create_dataset(
                                key, 
                                data=src_dataset[:], 
                                compression=src_dataset.compression,
                                compression_opts=src_dataset.compression_opts
                            )
                        
                        # Add raw audio data
                        seq_length = None
                        for key in obj.keys():
                            if len(obj[key].shape) > 0:
                                seq_length = obj[key].shape[0]
                                break
                        
                        if seq_length is not None:
                            if audio_data_generator:
                                raw_audio = audio_data_generator(seq_length)
                            else:
                                raw_audio = generate_raw_audio(seq_length, sample_rate=sample_rate, audio_length=audio_length)
                            
                            dst_file[name].create_dataset(
                                'audio',
                                data=raw_audio,
                                compression='gzip',
                                compression_opts=9
                            )
                            print(f"    Added raw audio with shape: {raw_audio.shape} (length: {audio_length}s, sample_rate: {sample_rate}Hz)")
                        else:
                            print(f"    Warning: Could not determine sequence length for {name}")
                
                elif isinstance(obj, h5py.Dataset):
                    # Copy dataset
                    if name not in dst_file:
                        dst_file.create_dataset(
                            name,
                            data=obj[:],
                            compression=obj.compression,
                            compression_opts=obj.compression_opts
                        )
                    
                    # Copy dataset attributes
                    for attr_key, attr_value in obj.attrs.items():
                        dst_file[name].attrs[attr_key] = attr_value
            
            # Copy all items
            src_file.visititems(copy_item)
            
            # Copy attributes
            for key, value in src_file.attrs.items():
                dst_file.attrs[key] = value
    
    print(f"Successfully created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Add audio data to HDF5 datasets')
    parser.add_argument('--input-dir', type=str, 
                       default='/home/robot/Dataset_and_Checkpoint/mimicsonic/datasets',
                       help='Input directory containing HDF5 files')
    parser.add_argument('--output-dir', type=str,
                       default='/home/robot/Dataset_and_Checkpoint/mimicsonic/datasets',
                       help='Output directory for new HDF5 files')
    parser.add_argument('--files', nargs='+', 
                       default=['timer_Oct10.hdf5', 'human_hand_test/mimic_dual_hand.hdf5'],
                       help='HDF5 files to process')
    parser.add_argument('--audio-length', type=float, default=2.0,
                       help='Audio length in seconds (default: 2.0)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Audio sample rate in Hz (default: 16000)')
    
    args = parser.parse_args()
    
    print("=== Adding Raw Audio Data to HDF5 Datasets ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files to process: {args.files}")
    print(f"Audio length: {args.audio_length} seconds")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Audio samples per frame: {int(args.sample_rate * args.audio_length)}")
    
    for file_path in args.files:
        input_path = os.path.join(args.input_dir, file_path)
        
        # Generate output filename
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_path)
        name, ext = os.path.splitext(file_name)
        output_file_name = f"{name}_audio{ext}"
        
        if file_dir:
            output_path = os.path.join(args.output_dir, file_dir, output_file_name)
        else:
            output_path = os.path.join(args.output_dir, output_file_name)
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            continue
        
        # Process the file
        try:
            copy_hdf5_with_audio(input_path, output_path, 
                               audio_length=args.audio_length, 
                               sample_rate=args.sample_rate)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            continue
    
    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()
