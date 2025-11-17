#!/usr/bin/env python3
"""
RealSense D415 Hand Data Recording Script with Audio Support and Mark Marking
Features:
1. Print camera intrinsics
2. Record RGB and depth video
3. Record audio simultaneously
4. Save as HDF5 format with audio data
5. Optional MP4 video generation
6. Support for multiple file recording
7. Keyboard controls for recording management
8. Mark marking functionality during recording

Usage:
Single file: python3 record_hand_rs_mask.py --output_dir ./recordings
Multiple files: python3 record_hand_rs_mask.py --output_dir ./recordings --start_idx 0 --end_idx 5
With video: python3 record_hand_rs_mask.py --output_dir ./recordings --generate_video

Controls:
- 'b': Start recording
- 'e': Stop recording and save
- 'm': Mark timestep (during recording)
- 'q': Quit program
- 'c': Continue to next file (multi-file mode, in camera window)
- 'r': Repeat current file (multi-file mode, in camera window)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import h5py
import argparse
import os
import json
from datetime import datetime
import time
import queue
import pyaudio
import wave
import threading

# Audio recording configuration
AUDIO_ENABLED = True  # Flag to enable/disable audio recording
AUDIO_SAMPLING_RATE = 48000
AUDIO_CHANNELS = 1
AUDIO_BUFFER_SIZE = 256
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_DEVICE_NAME = "USB PnP Audio Device"  # Target audio device name

def setup_audio_device():
    """Initialize and configure audio device for recording"""
    if not AUDIO_ENABLED:
        return None, None, None
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Find target audio device
    device_index = None
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if AUDIO_DEVICE_NAME in device_info["name"]:
            device_index = i
            print(f"Found audio device: {device_info['name']}, Index: {device_index}")
            break
    
    if device_index is None:
        print(f"Warning: Audio device '{AUDIO_DEVICE_NAME}' not found. Using default device.")
        device_index = None
    
    # Create audio stream
    stream = audio.open(
        format=AUDIO_FORMAT,
        channels=AUDIO_CHANNELS,
        rate=AUDIO_SAMPLING_RATE,
        input=True,
        frames_per_buffer=AUDIO_BUFFER_SIZE,
        input_device_index=device_index
    )
    
    return audio, stream, device_index

class RealSenseRecorder:
    def __init__(self, serial_number="109422062625", width=640, height=480, fps=30):
        """
        Initialize RealSense D415 recorder
        
        Args:
            serial_number: Camera serial number
            width: Image width
            height: Image height
            fps: Frame rate
        """
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize recorder
        self.pipeline = None
        self.config = None
        self.align = None
        
        # Recording state
        self.recording = False
        self.frames_rgb = []
        self.frames_depth = []
        self.timestamps = []
        self.frame_count = 0
        
        # Audio recording
        self.audio = None
        self.audio_stream = None
        self.audio_queue = queue.Queue()
        self.audio_recording = False
        self.audio_chunks = []
        
        # Mark marking functionality
        self.mark_timesteps = []
        self.last_mark_timestep = -1000
        self.MIN_MARK_DISTANCE = 100  # minimum distance between marks (frames)
        
    def setup_audio_recording(self):
        """Setup audio recording"""
        if AUDIO_ENABLED:
            self.audio, self.audio_stream, _ = setup_audio_device()
            if self.audio is not None:
                print("✅ Audio recording setup completed")
            else:
                print("❌ Audio recording setup failed")
        else:
            print("🔇 Audio recording disabled")
    
    def start_audio_recording(self):
        """Start audio recording in background thread"""
        if AUDIO_ENABLED and self.audio is not None:
            self.audio_recording = True
            self.audio_chunks = []
            
            def record_audio():
                """Background thread: continuously capture audio data and store in audio_queue"""
                while self.audio_recording:
                    try:
                        audio_block = self.audio_stream.read(AUDIO_BUFFER_SIZE, exception_on_overflow=False)
                        self.audio_queue.put(audio_block)
                    except Exception as e:
                        print(f"Audio recording error: {e}")
                        break
            
            self.audio_thread = threading.Thread(target=record_audio, daemon=True)
            self.audio_thread.start()
            print("🎤 Audio recording started...")
    
    def stop_audio_recording(self):
        """Stop audio recording"""
        if AUDIO_ENABLED and self.audio is not None:
            self.audio_recording = False
            if hasattr(self, 'audio_thread'):
                self.audio_thread.join()
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio.terminate()
            print("🎤 Audio recording stopped.")
    
    def collect_audio_chunks(self):
        """Collect audio chunks for current frame"""
        if not AUDIO_ENABLED or self.audio is None:
            return 0
            
        chunks_collected = 0
        while not self.audio_queue.empty():
            self.audio_chunks.append(self.audio_queue.get())
            chunks_collected += 1
        return chunks_collected
    
    def setup_camera(self):
        """Setup RealSense camera"""
        print(f"🔧 Setting up RealSense D415 (Serial: {self.serial_number})")
        
        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # Create align object
        self.align = rs.align(rs.stream.color)
        
        # Start streaming
        profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        
        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()
        
        print("📷 Camera intrinsics:")
        print(f"  Depth: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}, "
              f"ppx={depth_intrinsics.ppx:.2f}, ppy={depth_intrinsics.ppy:.2f}")
        print(f"  Color: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}, "
              f"ppx={color_intrinsics.ppx:.2f}, ppy={color_intrinsics.ppy:.2f}")
        
        return True
    
    def start_recording(self):
        """Start recording"""
        if not self.recording:
            self.recording = True
            self.frames_rgb = []
            self.frames_depth = []
            self.timestamps = []
            self.frame_count = 0
            self.audio_chunks = []
            # Reset mark tracking for new recording
            self.mark_timesteps = []
            self.last_mark_timestep = -1000
            
            # Clear audio queue to discard any pre-accumulated audio data
            if AUDIO_ENABLED and self.audio is not None:
                while not self.audio_queue.empty():
                    self.audio_queue.get()
                # print("🎤 Audio queue cleared before starting recording")
            
            print("🔴 Recording started")
            return True
        return False
    
    def stop_recording(self):
        """Stop recording"""
        if self.recording:
            self.recording = False
            print(f"⏹️  Stopped recording, total {self.frame_count} frames")
            return True
        return False
    
    def mark(self):
        """Mark current frame as mark timestep"""
        if self.recording:
            # Check if enough time has passed since last mark
            if self.frame_count - self.last_mark_timestep >= self.MIN_MARK_DISTANCE:
                self.mark_timesteps.append(self.frame_count)
                self.last_mark_timestep = self.frame_count
                print(f"🎯 Mark marked at frame {self.frame_count}")
                return True
            else:
                remaining_distance = self.MIN_MARK_DISTANCE - (self.frame_count - self.last_mark_timestep)
                print(f"⚠️ Mark request ignored. Need {remaining_distance} more frames between marks.")
                return False
        return False
    
    def process_frame(self):
        """Process single frame"""
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth to color
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        timestamp = time.time()
        
        # If recording, save data
        if self.recording:
            # Convert BGR to RGB before saving
            color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            self.frames_rgb.append(color_rgb.copy())
            self.frames_depth.append(depth_image.copy())
            self.timestamps.append(timestamp)
            self.frame_count += 1
            
            # Collect audio chunks during recording
            if AUDIO_ENABLED and self.audio_recording:
                chunks_collected = self.collect_audio_chunks()
                if chunks_collected > 0 and self.frame_count < 5:  # Debug info for first 5 frames
                    print(f"Frame {self.frame_count}: collected {chunks_collected} audio chunks")
            
            # Print progress every second
            if self.frame_count % self.fps == 0:
                print(f"📹 Recording: {self.frame_count / self.fps:.0f} seconds")
        
        return color_image, depth_image, timestamp
        
    def visualize_stream(self, color_image, depth_image):
        """Visualize data stream"""
        # Depth image visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Add recording status indicator
        if self.recording:
            cv2.circle(color_image, (30, 30), 10, (0, 0, 255), -1)  # Red circle
            cv2.putText(color_image, f"REC {self.frame_count}", (50, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.circle(color_image, (30, 30), 10, (0, 255, 0), -1)  # Green circle
            cv2.putText(color_image, "READY", (50, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add mark count indicator
        if len(self.mark_timesteps) > 0:
            cv2.putText(color_image, f"MARKS: {len(self.mark_timesteps)}", (10, color_image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Combine images
        display_image = np.hstack((color_image, depth_colormap))
        
        return display_image
    
    def save_data(self, output_path):
        """Save recorded data to HDF5"""
        if len(self.frames_rgb) == 0:
            print("❌ No data to save")
            return False
        
        print(f"💾 Saving data to {output_path}")
        
        try:
            with h5py.File(output_path, 'w') as f:
                # Save video data
                f.create_dataset('rgb', data=np.array(self.frames_rgb), compression='gzip')
                f.create_dataset('depth', data=np.array(self.frames_depth), compression='gzip')
                f.create_dataset('timestamps', data=np.array(self.timestamps))
                
                # Save metadata
                f.attrs['fps'] = self.fps
                f.attrs['width'] = self.width
                f.attrs['height'] = self.height
                f.attrs['total_frames'] = len(self.frames_rgb)
                f.attrs['serial_number'] = self.serial_number
                f.attrs['recording_time'] = datetime.now().isoformat()
                
                # Save mark timesteps if any
                if len(self.mark_timesteps) > 0:
                    f.create_dataset('mark_timesteps', data=np.array(self.mark_timesteps, dtype=np.int32))
                    f.attrs['num_marks'] = len(self.mark_timesteps)
                    print(f"🎯 Saved {len(self.mark_timesteps)} mark timesteps: {self.mark_timesteps}")
                else:
                    f.attrs['num_marks'] = 0
                    print("🎯 No marks were marked during this recording")
                
                # Save audio data if available
                if AUDIO_ENABLED and len(self.audio_chunks) > 0:
                    max_substeps = 2048  # Maximum audio samples per frame
                    audio_chunks_np = np.zeros((len(self.frames_rgb), max_substeps), dtype=np.int16)
                    
                    # Distribute audio chunks across frames
                    chunks_per_frame = len(self.audio_chunks) // len(self.frames_rgb)
                    for i in range(len(self.frames_rgb)):
                        start_chunk = i * chunks_per_frame
                        end_chunk = start_chunk + chunks_per_frame
                        if i == len(self.frames_rgb) - 1:  # Last frame gets remaining chunks
                            end_chunk = len(self.audio_chunks)
                        
                        if start_chunk < len(self.audio_chunks):
                            frame_audio_chunks = self.audio_chunks[start_chunk:end_chunk]
                            if frame_audio_chunks:
                                audio_data = np.frombuffer(b"".join(frame_audio_chunks), dtype=np.int16)
                                if len(audio_data) > max_substeps - 1:
                                    audio_data = audio_data[:max_substeps-1]
                                audio_chunks_np[i, :len(audio_data)] = audio_data
                                audio_chunks_np[i, -1] = len(audio_data)  # Store actual length in last element
                    
                    f.create_dataset("audio", data=audio_chunks_np, dtype=np.int16)
                    f.attrs['audio_sampling_rate'] = AUDIO_SAMPLING_RATE
                    f.attrs['audio_channels'] = AUDIO_CHANNELS
                    print(f"🎤 Audio data saved: {len(self.frames_rgb)} frames, max {max_substeps} samples per frame")
                else:
                    print("🔇 No audio data to save")
            
            print(f"✅ Data saved successfully: {len(self.frames_rgb)} frames")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save data: {e}")
            return False
    
    def generate_video(self, hdf5_path, fps=None):
        """Generate video files from HDF5"""
        if fps is None:
            fps = self.fps

        print("🎥 Generating video files...")

        try:
            with h5py.File(hdf5_path, 'r') as f:
                rgb_frames = f['rgb'][:]          # Stored as RGB now
                depth_frames = f['depth'][:]

            if rgb_frames.shape[0] == 0:
                print("❌ No frame data in HDF5 file")
                return False

            height, width, _ = rgb_frames[0].shape

            base_name = os.path.splitext(hdf5_path)[0]
            rgb_video_path = f"{base_name}_rgb.mp4"
            depth_video_path = f"{base_name}_depth.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_rgb = cv2.VideoWriter(rgb_video_path, fourcc, fps, (width, height))

            if not out_rgb.isOpened():
                print(f"❌ Cannot create RGB video writer: {rgb_video_path}")
                return False

            # Write frames
            for i in range(rgb_frames.shape[0]):
                out_rgb.write(rgb_frames[i])
                
                # Normalize depth image for visualization
                depth_normalized = cv2.normalize(
                    depth_frames[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                )
                out_depth.write(depth_normalized)
                
            out_rgb.release()
            out_depth.release()
            
            print(f"✅ RGB video saved to: {rgb_video_path}")
            print(f"✅ Depth video saved to: {depth_video_path}")
            return True
            
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.pipeline:
            self.pipeline.stop()
        if AUDIO_ENABLED and self.audio is not None:
            self.stop_audio_recording()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='RealSense D415 Hand Data Recording with Mark Support')
    parser.add_argument('--output_dir', type=str, 
        default="./recordings",
        help='Output directory for recordings')
    parser.add_argument('--start_idx', type=int, default=0,
        help='Start index for multi-file recording')
    parser.add_argument('--end_idx', type=int, default=0,
        help='End index for multi-file recording')
    parser.add_argument('--generate_video', action='store_true',
        help='Generate MP4 videos after recording')
    parser.add_argument('--serial_number', type=str, default="109422062625",
        help='Camera serial number')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine recording mode
    multi_file_mode = args.start_idx != args.end_idx
    
    if multi_file_mode:
        print(f"📁 Multi-file mode: {args.start_idx} to {args.end_idx}")
        current_idx = args.start_idx
        end_idx = args.end_idx
    else:
        print("📁 Single file mode")
        current_idx = args.start_idx
        end_idx = args.start_idx
    
    # Initialize recorder
    recorder = RealSenseRecorder(serial_number=args.serial_number)
    
    try:
        # Setup camera and audio
        if not recorder.setup_camera():
            print("❌ Failed to setup camera")
            return
        
        recorder.setup_audio_recording()
        recorder.start_audio_recording()
        
        print("\n🎮 Controls:")
        print("  'b': Start recording")
        print("  'e': Stop recording and save")
        print("  'm': Mark timestep (during recording)")
        print("  'q': Quit program")
        if multi_file_mode:
            print("  'c': Continue to next file (in camera window)")
            print("  'r': Repeat current file (in camera window)")
        
        # Main recording loop
        while current_idx <= end_idx:
            print(f"\n📹 File {current_idx}/{end_idx}")
            
            # Generate filename
            if args.start_idx == args.end_idx == 0:
                # Single unindexed recording - use timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"realsense_hand_data_{timestamp}.hdf5"
            else:
                # Indexed recording
                filename = f"realsense_hand_data_{current_idx:03d}.hdf5"
            
            output_path = os.path.join(args.output_dir, filename)
            
            print(f"📁 Output: {output_path}")
            print("🎬 Press 'b' to start recording...")
            
            # Recording session loop
            try:
                while True:
                    # Process frame
                    color_image, depth_image, timestamp = recorder.process_frame()
                    if color_image is None:
                        continue
                    
                    # Visualize
                    display_image = recorder.visualize_stream(color_image, depth_image)
                    
                    # Add file info for multi-file mode
                    if multi_file_mode:
                        status_text = f"File {current_idx}/{end_idx}"
                        cv2.putText(display_image, status_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('RealSense D415 Recorder with Mark Support', display_image)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('b'):
                        recorder.start_recording()
                    elif key == ord('e'):
                        if recorder.stop_recording():
                            # Save data
                            if recorder.save_data(output_path):
                                print(f"✅ Recording saved: {output_path}")
                                
                                # Generate video if requested
                                if args.generate_video:
                                    recorder.generate_video(output_path)
                            else:
                                print("❌ Failed to save recording")
                            break
                    elif key == ord('m'):
                        recorder.mark()
                    elif key == ord('q'):
                        print("👋 Exiting program")
                        return
                    elif multi_file_mode and key == ord('c') and current_idx < end_idx:
                        print("⏭️  Moving to next file...")
                        break
                    elif multi_file_mode and key == ord('r'):
                        print("🔄 Repeating current file...")
                        break
                        
            except KeyboardInterrupt:
                print("\n⚠️  User interrupted")
            except Exception as e:
                print(f"❌ Error: {e}")
            finally:
                # Stop recording if still active
                if recorder.recording:
                    recorder.stop_recording()
            
            # Move to next file in multi-file mode
            if multi_file_mode:
                current_idx += 1
                
                # Wait for user input to continue (including after last file)
                if current_idx <= end_idx:
                    print(f"\n⏸️  File {current_idx-1} completed.")
                    print("Press 'c' to continue to next file, 'r' to repeat, 'q' to quit:")
                else:
                    # Last file completed
                    print(f"\n⏸️  File {current_idx-1} completed.")
                    print("Press 'c' to finish all recordings, 'r' to repeat last file, 'q' to quit:")
                
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('c'):
                        break
                    elif key == ord('r'):
                        current_idx -= 1
                        break
                    elif key == ord('q'):
                        print("👋 Exiting program")
                        return
                    else:
                        if current_idx <= end_idx:
                            print("Invalid input. Press 'c' to continue, 'r' to repeat, 'q' to quit.")
                        else:
                            print("Invalid input. Press 'c' to finish, 'r' to repeat, 'q' to quit.")
            else:
                break
        
        print("✅ All recordings completed!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    main()