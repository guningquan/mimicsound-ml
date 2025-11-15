#!/usr/bin/env python3
"""
Script to synthesize MP4 video with audio from HDF5 files
Combines RGB video and audio data from RealSense recordings
"""

import h5py
import numpy as np
import cv2
import argparse
import os
from scipy.io import wavfile
import tempfile

def synthesize_video_from_hdf5(hdf5_path, output_path=None, fps=None):
    """
    Synthesize MP4 video with audio from HDF5 file
    
    Args:
        hdf5_path: Path to input HDF5 file
        output_path: Path to output MP4 file (optional)
        fps: Frame rate for output video (optional)
    """
    
    if not os.path.exists(hdf5_path):
        print(f"❌ File not found: {hdf5_path}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(hdf5_path)[0]
        output_path = f"{base_name}_synthesized.mp4"
    
    print(f"📁 Reading HDF5 file: {hdf5_path}")
    print(f"📹 Output video: {output_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check if required data exists
            if 'rgb' not in f:
                print("❌ No RGB data found in HDF5 file")
                return False
            
            # Audio is optional - video can be generated without audio
            has_audio = 'audio' in f
            
            # Read RGB data
            rgb_data = f['rgb'][:]
            print(f"🎥 RGB data shape: {rgb_data.shape}")
            
            # Read audio data if available
            audio_data_raw = None
            if has_audio:
                audio_data_raw = f['audio'][:]  # Format: (num_timesteps, max_samples), last column stores actual length
                print(f"🎤 Audio data shape: {audio_data_raw.shape}")
            else:
                print("⚠️  No audio data found in HDF5 file, will create video without audio")
            
            # Get metadata
            if 'fps' in f.attrs:
                original_fps = f.attrs['fps']
                print(f"📊 Original FPS: {original_fps}")
            else:
                original_fps = 30  # Default FPS
            
            if 'audio_sampling_rate' in f.attrs:
                audio_sampling_rate = f.attrs['audio_sampling_rate']
                print(f"🎵 Audio sampling rate: {audio_sampling_rate}")
            else:
                audio_sampling_rate = 48000  # Default sampling rate
            
            # Use provided fps or original fps
            if fps is None:
                fps = original_fps
            
            print(f"🎬 Using FPS: {fps}")
            
            # Get video dimensions
            num_frames, height, width, channels = rgb_data.shape
            print(f"📐 Video dimensions: {width}x{height}, {num_frames} frames")
            
            # Extract and concatenate audio data from 2D format
            # Format: (num_timesteps, max_samples), last column stores actual length
            audio_data = None
            temp_audio_path = None
            
            if audio_data_raw is not None:
                print("🎵 Extracting audio data from timesteps...")
                concatenated_audio = []
                
                if len(audio_data_raw.shape) == 2:
                    # Format: (num_timesteps, max_samples) - last column stores actual length
                    for t in range(audio_data_raw.shape[0]):
                        # Get actual length (stored in last element)
                        actual_length = int(audio_data_raw[t, -1])
                        if actual_length > 0:
                            # Extract valid audio data (excluding the length element)
                            timestep_audio = audio_data_raw[t, :actual_length]
                            concatenated_audio.append(timestep_audio)
                        # else: no audio data for this timestep, skip it
                    
                    if not concatenated_audio:
                        print("⚠️  No valid audio data found, creating video without audio")
                        audio_data = None
                        temp_audio_path = None
                    else:
                        # Concatenate all timesteps into 1D array
                        full_audio = np.concatenate(concatenated_audio)
                        print(f"🎵 Total audio samples: {len(full_audio)}")
                        print(f"🎵 Audio duration: {len(full_audio) / audio_sampling_rate:.2f} seconds")
                        
                        # Create temporary audio file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                            temp_audio_path = temp_audio.name
                            
                            # Convert audio data to proper format
                            # Audio data is int16, normalize to [-1, 1] range for wavfile
                            if full_audio.dtype == np.int16:
                                audio_normalized = full_audio.astype(np.float32) / 32768.0
                            else:
                                audio_normalized = full_audio.astype(np.float32)
                            
                            # Ensure audio is 1D array (mono channel)
                            if len(audio_normalized.shape) > 1:
                                audio_normalized = audio_normalized.flatten()
                            
                            # Write temporary audio file (mono, 1 channel)
                            wavfile.write(temp_audio_path, audio_sampling_rate, audio_normalized)
                            print(f"🎵 Created temporary audio file: {temp_audio_path}")
                            audio_data = audio_normalized
                else:
                    # Already 1D audio data (unusual format, handle gracefully)
                    print("⚠️  Unexpected audio format (not 2D), attempting to process as 1D...")
                    full_audio = audio_data_raw.flatten()
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                        if full_audio.dtype == np.int16:
                            audio_normalized = full_audio.astype(np.float32) / 32768.0
                        else:
                            audio_normalized = full_audio.astype(np.float32)
                        wavfile.write(temp_audio_path, audio_sampling_rate, audio_normalized)
                        print(f"🎵 Created temporary audio file: {temp_audio_path}")
                        audio_data = audio_normalized
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("❌ Failed to create video writer")
                os.unlink(temp_audio_path)  # Clean up temp file
                return False
            
            print("🎬 Writing video frames...")
            
            # Write video frames
            for i in range(num_frames):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(rgb_data[i], cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / num_frames * 100
                    print(f"  Progress: {progress:.1f}% ({i + 1}/{num_frames} frames)")
            
            out.release()
            print("✅ Video writing completed")
            
            # Combine video and audio if audio data is available
            if audio_data is not None and temp_audio_path is not None:
                # Now we need to combine video and audio
                # This requires ffmpeg or similar tool
                print("🔗 Combining video and audio...")
                
                # Create final output path with audio
                final_output_path = output_path.replace('.mp4', '_with_audio.mp4')
                
                # Use ffmpeg to combine video and audio
                import subprocess
                
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output file
                    '-i', output_path,  # Input video
                    '-i', temp_audio_path,  # Input audio
                    '-c:v', 'copy',  # Copy video codec
                    '-c:a', 'aac',  # Use AAC audio codec
                    '-shortest',  # End when shortest input ends
                    final_output_path
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(f"✅ Video with audio created: {final_output_path}")
                    
                    # Clean up temporary files
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    if os.path.exists(output_path):
                        os.unlink(output_path)  # Remove video-only file
                    
                    # Rename final file to original output path
                    os.rename(final_output_path, output_path)
                    print(f"✅ Final video saved as: {output_path}")
                    
                    return True
                    
                except subprocess.CalledProcessError as e:
                    print(f"❌ FFmpeg error: {e}")
                    print(f"FFmpeg stderr: {e.stderr}")
                    print("💡 Make sure ffmpeg is installed: sudo apt install ffmpeg")
                    
                    # Clean up temporary files
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    if os.path.exists(final_output_path):
                        os.unlink(final_output_path)
                    
                    # Keep video-only file if audio combination fails
                    print(f"⚠️  Keeping video-only file: {output_path}")
                    return False
                except FileNotFoundError:
                    print("❌ FFmpeg not found. Please install ffmpeg:")
                    print("   sudo apt install ffmpeg")
                    
                    # Clean up temporary files
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    if os.path.exists(final_output_path):
                        os.unlink(final_output_path)
                    
                    # Keep video-only file if ffmpeg not found
                    print(f"⚠️  Keeping video-only file: {output_path}")
                    return False
            else:
                # No audio data, video is already complete
                print("✅ Video saved (without audio)")
                return True
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Synthesize MP4 video with audio from HDF5 files")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input HDF5 file path"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output MP4 file path (optional)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        help="Output video frame rate (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Synthesize video
    success = synthesize_video_from_hdf5(args.input, args.output, args.fps)
    
    if success:
        print("🎉 Video synthesis completed successfully!")
    else:
        print("❌ Video synthesis failed!")

if __name__ == "__main__":
    main()