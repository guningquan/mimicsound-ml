#!/usr/bin/env python3
"""
RealSense D415 Hand Data Recording Script with Audio Support
Features:
1. Print camera intrinsics
2. Record RGB and depth video
3. Record audio simultaneously
4. Save as HDF5 format with audio data
5. Optional MP4 video generation
6. Support for multiple file recording
7. Keyboard controls for recording management

Usage:
Single file: python3 record_hand_rs.py --output_dir ./recordings
Multiple files: python3 record_hand_rs.py --output_dir ./recordings --start_idx 0 --end_idx 5
With video: python3 record_hand_rs.py --output_dir ./recordings --generate_video

Controls:
- 'c': Start recording
- 'e': Stop recording and save
- 'q': Quit program
- 'n': Next file (multi-file mode, in camera window)
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
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.pipeline_started = False
        
        # Recording data
        self.frames_rgb = []
        self.frames_depth = []
        self.timestamps = []
        self.recording = False
        self.frame_count = 0
        
        # Audio recording data
        self.audio_chunks = []
        self.audio_queue = queue.Queue()
        self.audio_recording = False
        self.audio_recording_thread = None
        self.audio = None
        self.audio_stream = None
    
    def setup_audio_recording(self):
        """Setup audio recording for this instance"""
        if not AUDIO_ENABLED:
            return False
        
        try:
            self.audio, self.audio_stream, _ = setup_audio_device()
            if self.audio is not None:
                print("Audio recording setup completed!")
                return True
            else:
                print("Audio recording setup failed!")
                return False
        except Exception as e:
            print(f"Audio setup error: {e}")
            return False
    
    def start_audio_recording(self):
        """Start audio recording thread"""
        if not AUDIO_ENABLED or self.audio is None:
            return False
        
        if not self.audio_recording:
            self.audio_recording = True
            self.audio_chunks = []
            
            def record_audio():
                """Background thread: continuously capture audio data"""
                while self.audio_recording:
                    try:
                        audio_block = self.audio_stream.read(AUDIO_BUFFER_SIZE, exception_on_overflow=False)
                        self.audio_queue.put(audio_block)
                    except Exception as e:
                        print(f"Audio recording error: {e}")
                        break
            
            self.audio_recording_thread = threading.Thread(target=record_audio, daemon=True)
            self.audio_recording_thread.start()
            print("🎤 Audio recording started...")
            return True
        return False
    
    def stop_audio_recording(self):
        """Stop audio recording thread"""
        if self.audio_recording:
            self.audio_recording = False
            if self.audio_recording_thread is not None:
                self.audio_recording_thread.join()
            print("🎤 Audio recording stopped...")
            return True
        return False
    
    def collect_audio_chunks(self):
        """Collect audio chunks from queue"""
        chunks_collected = 0
        while not self.audio_queue.empty():
            self.audio_chunks.append(self.audio_queue.get())
            chunks_collected += 1
        return chunks_collected
    
    def cleanup_audio(self):
        """Cleanup audio resources"""
        if self.audio_recording:
            self.stop_audio_recording()
        
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        
        if self.audio is not None:
            try:
                self.audio.terminate()
            except:
                pass
        
    def check_camera_availability(self):
        """Check if camera is available"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("❌ No RealSense devices found")
                return False
                
            for device in devices:
                if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                    print(f"✅ Found camera with serial number: {self.serial_number}")
                    return True
                    
            print(f"❌ Camera with serial number {self.serial_number} not found")
            print("Available cameras:")
            for device in devices:
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                print(f"  - Serial: {serial}, Name: {name}")
            return False
            
        except Exception as e:
            print(f"❌ Error checking camera availability: {e}")
            return False
    
    def check_supported_configs(self):
        """Check supported stream configurations"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for device in devices:
                if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                    # print("Supported stream configurations:")
                    
                    # Check depth stream profiles
                    depth_profiles = device.sensors[0].get_stream_profiles()
                    # print("  Depth streams:")
                    for profile in depth_profiles:
                        if profile.stream_type() == rs.stream.depth:
                            video_profile = profile.as_video_stream_profile()
                            # print(f"    {video_profile.width()}x{video_profile.height()}@{video_profile.fps()}fps")
                    
                    # Check color stream profiles
                    color_profiles = device.sensors[1].get_stream_profiles()
                    # print("  Color streams:")
                    for profile in color_profiles:
                        if profile.stream_type() == rs.stream.color:
                            video_profile = profile.as_video_stream_profile()
                            # print(f"    {video_profile.width()}x{video_profile.height()}@{video_profile.fps()}fps")
                    return True
            return False
            
        except Exception as e:
            print(f"❌ Error checking supported configs: {e}")
            return False
    
    def find_best_config(self):
        """Find the best supported configuration"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for device in devices:
                if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                    # Get all stream profiles
                    depth_profiles = []
                    color_profiles = []
                    
                    for sensor in device.sensors:
                        for profile in sensor.get_stream_profiles():
                            if profile.stream_type() == rs.stream.depth:
                                depth_profiles.append(profile.as_video_stream_profile())
                            elif profile.stream_type() == rs.stream.color:
                                color_profiles.append(profile.as_video_stream_profile())
                    
                    # Find matching configurations
                    for depth_profile in depth_profiles:
                        for color_profile in color_profiles:
                            if (depth_profile.width() == color_profile.width() and 
                                depth_profile.height() == color_profile.height() and
                                depth_profile.fps() == color_profile.fps()):
                                
                                # Prefer 640x480@30fps for compatibility
                                if (depth_profile.width() == 640 and 
                                    depth_profile.height() == 480 and 
                                    depth_profile.fps() == 30):
                                    return depth_profile.width(), depth_profile.height(), depth_profile.fps()
                    
                    # If no exact match, use the first available depth profile
                    if depth_profiles:
                        profile = depth_profiles[0]
                        return profile.width(), profile.height(), profile.fps()
            
            # Fallback to safe defaults
            return 640, 480, 30
            
        except Exception as e:
            print(f"❌ Error finding best config: {e}")
            return 640, 480, 30
        
    def initialize_camera(self):
        """Initialize camera"""
        print("Initializing RealSense D415 camera...")
        
        try:
            # Find the best supported configuration
            width, height, fps = self.find_best_config()
            
            print(f"Using configuration: {width}x{height}@{fps}fps")
            
            # Configure streams
            self.config = rs.config()
            self.config.enable_device(self.serial_number)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            
            # Start pipeline
            self.pipeline = rs.pipeline()
            profile = self.pipeline.start(self.config)
            self.pipeline_started = True
            
            # Get stream profiles and intrinsics
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            
            self.depth_intrinsics = depth_profile.get_intrinsics()
            self.color_intrinsics = color_profile.get_intrinsics()
            
            # Create align object (align depth to RGB)
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            print("Camera initialization completed!")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            print("Please check:")
            print("1. Camera is properly connected")
            print("2. Camera serial number is correct")
            print("3. No other programs are using the camera")
            print("4. RealSense SDK is properly installed")
            raise
        
    def print_intrinsics(self):
        """Print camera intrinsics"""
        print("\n" + "="*50)
        print("RealSense D415 Camera Intrinsics")
        print("="*50)
        print(f"Serial Number: {self.serial_number}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Frame Rate: {self.fps} FPS")
        print()
        
        # Print RGB camera intrinsics
        if self.color_intrinsics:
            print("RGB Camera Intrinsics:")
            print(f"  fx: {self.color_intrinsics.fx:.6f}")
            print(f"  fy: {self.color_intrinsics.fy:.6f}")
            print(f"  ppx: {self.color_intrinsics.ppx:.6f}")
            print(f"  ppy: {self.color_intrinsics.ppy:.6f}")
            print(f"  model: {self.color_intrinsics.model}")
            print(f"  coeffs: {list(self.color_intrinsics.coeffs)}")
            print()
            
            # Build and print intrinsics matrix
            rgb_matrix = self.build_intrinsics_matrix(self.color_intrinsics)
            print("RGB Camera Intrinsics Matrix (3x4):")
            print("REALSENSE_INTRINSICS = np.array(")
            print(f"    [[{rgb_matrix[0,0]:.1f}, {rgb_matrix[0,1]:.1f}, {rgb_matrix[0,2]:.1f}, {rgb_matrix[0,3]:.1f}],")
            print(f"     [{rgb_matrix[1,0]:.1f}, {rgb_matrix[1,1]:.1f}, {rgb_matrix[1,2]:.1f}, {rgb_matrix[1,3]:.1f}],")
            print(f"     [{rgb_matrix[2,0]:.1f}, {rgb_matrix[2,1]:.1f}, {rgb_matrix[2,2]:.1f}, {rgb_matrix[2,3]:.1f}]]")
            print(")")
            print()
        
        # Print depth camera intrinsics
        # if self.depth_intrinsics:
        #     print("Depth Camera Intrinsics:")
        #     print(f"  fx: {self.depth_intrinsics.fx:.6f}")
        #     print(f"  fy: {self.depth_intrinsics.fy:.6f}")
        #     print(f"  ppx: {self.depth_intrinsics.ppx:.6f}")
        #     print(f"  ppy: {self.depth_intrinsics.ppy:.6f}")
        #     print(f"  model: {self.depth_intrinsics.model}")
        #     print(f"  coeffs: {list(self.depth_intrinsics.coeffs)}")
        #     print()
            
            # Build and print depth intrinsics matrix
            depth_matrix = self.build_intrinsics_matrix(self.depth_intrinsics)
            # print("Depth Camera Intrinsics Matrix (3x4):")
            # print("DEPTH_INTRINSICS = np.array(")
            # print(f"    [[{depth_matrix[0,0]:.1f}, {depth_matrix[0,1]:.1f}, {depth_matrix[0,2]:.1f}, {depth_matrix[0,3]:.1f}],")
            # print(f"     [{depth_matrix[1,0]:.1f}, {depth_matrix[1,1]:.1f}, {depth_matrix[1,2]:.1f}, {depth_matrix[1,3]:.1f}],")
            # print(f"     [{depth_matrix[2,0]:.1f}, {depth_matrix[2,1]:.1f}, {depth_matrix[2,2]:.1f}, {depth_matrix[2,3]:.1f}]]")
            # print(")")
            # print()
        
        
    def get_intrinsics_dict(self, intrinsics):
        """Convert intrinsics object to dictionary"""
        return {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "model": str(intrinsics.model),
            "coeffs": list(intrinsics.coeffs)
        }
        
    def build_intrinsics_matrix(self, intrinsics):
        """Build 3x4 intrinsics matrix from intrinsics object"""
        return np.array([
            [intrinsics.fx, 0.0, intrinsics.ppx, 0.0],
            [0.0, intrinsics.fy, intrinsics.ppy, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
    def start_recording(self):
        """Start recording"""
        if not self.recording:
            self.recording = True
            self.frames_rgb = []
            self.frames_depth = []
            self.timestamps = []
            self.frame_count = 0
            
            # Start audio recording if available
            if AUDIO_ENABLED and self.audio is not None:
                self.start_audio_recording()
            
            print("🎬 Starting recording...")
        else:
            print("⚠️  Already recording")
            
    def stop_recording(self):
        """Stop recording"""
        if self.recording:
            self.recording = False
            
            # Stop audio recording if active
            if AUDIO_ENABLED and self.audio_recording:
                self.stop_audio_recording()
            
            print(f"⏹️  Stopped recording, total {self.frame_count} frames")
            return True
        else:
            print("⚠️  Not currently recording")
            return False
            
    def process_frame(self):
        """Process one frame"""
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth to RGB
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
            
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get timestamp
        timestamp = frames.get_timestamp()
        
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
            cv2.putText(color_image, "REC", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(depth_colormap, "REC", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Horizontal concatenation for display
        images = np.hstack((color_image, depth_colormap))
        
        # Add control instructions
        cv2.putText(images, "Press 'c' to start, 'e' to stop, 'q' to quit", 
                   (10, images.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return images
        
    def save_to_hdf5(self, output_path):
        """Save data to HDF5 file"""
        if not self.frames_rgb:
            print("❌ No recording data to save")
            return False
            
        print(f"💾 Saving data to {output_path}...")
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            with h5py.File(output_path, 'w') as f:
                # Save image data
                f.create_dataset('rgb', data=np.array(self.frames_rgb), compression="gzip")
                f.create_dataset('depth', data=np.array(self.frames_depth), compression="gzip")
                f.create_dataset('timestamps', data=np.array(self.timestamps), compression="gzip")
                
                # Save audio data if available
                if AUDIO_ENABLED and self.audio_chunks:
                    # Convert audio chunks to numpy array
                    audio_data = np.frombuffer(b"".join(self.audio_chunks), dtype=np.int16)
                    f.create_dataset('audio', data=audio_data, compression="gzip")
                    f.attrs['audio_sampling_rate'] = AUDIO_SAMPLING_RATE
                    f.attrs['audio_channels'] = AUDIO_CHANNELS
                    f.attrs['audio_duration'] = len(audio_data) / AUDIO_SAMPLING_RATE
                    print(f"🎤 Audio data saved: {len(audio_data)} samples ({len(audio_data) / AUDIO_SAMPLING_RATE:.2f}s)")
                
                # Save intrinsics
                # f.attrs['color_intrinsics'] = json.dumps(self.get_intrinsics_dict(self.color_intrinsics))
                # f.attrs['depth_intrinsics'] = json.dumps(self.get_intrinsics_dict(self.depth_intrinsics))
                # f.attrs['serial_number'] = self.serial_number
                f.attrs['resolution'] = f"{self.width}x{self.height}"
                f.attrs['fps'] = self.fps
                f.attrs['total_frames'] = len(self.frames_rgb)
                # f.attrs['recording_time'] = datetime.now().isoformat()
                
            print(f"✅ Data saved successfully! Total {len(self.frames_rgb)} frames")
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False
            
    # def generate_video(self, hdf5_path, fps=None):
        """Generate video files from HDF5"""
        if fps is None:
            fps = self.fps
            
        print("🎥 Generating video files...")
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                rgb_frames = f['rgb'][:]
                depth_frames = f['depth'][:]
                
            if rgb_frames.shape[0] == 0:
                print("❌ No frame data in HDF5 file")
                return False
                
            height, width, _ = rgb_frames[0].shape
            
            # Generate output filenames
            base_name = os.path.splitext(hdf5_path)[0]
            rgb_video_path = f"{base_name}_rgb.mp4"
            depth_video_path = f"{base_name}_depth.mp4"
            
            # RGB video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_rgb = cv2.VideoWriter(rgb_video_path, fourcc, fps, (width, height))
            
            if not out_rgb.isOpened():
                print(f"❌ Cannot create RGB video writer: {rgb_video_path}")
                return False
                
            # Depth video writer
            out_depth = cv2.VideoWriter(depth_video_path, fourcc, fps, (width, height), isColor=False)
            
            if not out_depth.isOpened():
                print(f"❌ Cannot create depth video writer: {depth_video_path}")
                out_rgb.release()
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

            out_depth = cv2.VideoWriter(depth_video_path, fourcc, fps, (width, height), isColor=False)
            if not out_depth.isOpened():
                print(f"❌ Cannot create depth video writer: {depth_video_path}")
                out_rgb.release()
                return False

            # --- Write frames ---
            for i in range(rgb_frames.shape[0]):
                # Convert RGB (stored) -> BGR (OpenCV writer expects BGR)
                frame_bgr = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2BGR)  # <-- key fix
                out_rgb.write(frame_bgr)

                # Normalize depth to 8-bit for writing as grayscale video
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

    def run(self, output_path, generate_video=False, multi_file_mode=False, current_idx=0, end_idx=0):
        """Run recording program"""
        try:
            # Check camera availability first
            if not self.check_camera_availability():
                print("❌ Camera not available, exiting...")
                return
            
            # Check supported configurations
            self.check_supported_configs()
                
            # Initialize camera
            self.initialize_camera()
            
            # Setup audio recording
            self.setup_audio_recording()
            
            # Print intrinsics
            self.print_intrinsics()
            
            print("\n🎮 Control Instructions:")
            print("  'c' - Start recording")
            print("  'e' - Stop recording and save")
            print("  'q' - Quit program")
            if multi_file_mode:
                print(f"  'n' - Next file (if not last file {current_idx}/{end_idx})")
                print(f"  'r' - Repeat current file")
            print("\nPress any key to start...")
            
            # Main loop
            while True:
                # Process frame
                color_image, depth_image, timestamp = self.process_frame()
                
                if color_image is None:
                    continue
                    
                # Visualize
                display_image = self.visualize_stream(color_image, depth_image)
                
                # Add multi-file mode status to display
                if multi_file_mode:
                    status_text = f"File {current_idx}/{end_idx}"
                    cv2.putText(display_image, status_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('RealSense D415 Recorder', display_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    self.start_recording()
                elif key == ord('e'):
                    if self.stop_recording():
                        break
                elif key == ord('q'):
                    print("👋 Exiting program")
                    break
                elif multi_file_mode and key == ord('n') and current_idx < end_idx:
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
            # Save data before cleanup
            if self.frames_rgb:
                if self.save_to_hdf5(output_path):
                    if generate_video:
                        self.generate_video(output_path)
            else:
                print("⚠️  No recording data")
            
            # Don't destroy windows in multi-file mode to keep display open
            if not multi_file_mode:
                # Clean up resources
                if self.pipeline and self.pipeline_started:
                    try:
                        self.pipeline.stop()
                    except RuntimeError as e:
                        print(f"Warning: Failed to stop pipeline: {e}")
                self.cleanup_audio()
                cv2.destroyAllWindows()


def get_auto_index(output_dir, prefix="realsense_hand_data_", suffix=".hdf5"):
    """Get next available file index"""
    max_idx = 1000
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for i in range(max_idx + 1):
        filename = f"{prefix}{i:03d}{suffix}"
        if not os.path.isfile(os.path.join(output_dir, filename)):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RealSense D415 Hand Data Recording Tool")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./recordings",
        help="Output directory for HDF5 files"
    )
    
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for file naming"
    )
    
    parser.add_argument(
        "--end_idx",
        type=int,
        default=0,
        help="End index for file naming (0 means single file)"
    )
    
    parser.add_argument(
        "--serial_number",
        type=str,
        default="109422062625",
        help="RealSense camera serial number"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate"
    )
    
    parser.add_argument(
        "--generate_video",
        action="store_true",
        help="Generate MP4 video files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create recorder
    recorder = RealSenseRecorder(
        serial_number=args.serial_number,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    # # Determine recording mode
    # if args.end_idx == 0 and args.start_idx == 0:
    #     # Single file mode with timestamp
    #     output_path = os.path.join(args.output_dir, f"realsense_hand_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5")
    #     recorder.run(output_path, args.generate_video)
    # else:
    # Multiple files mode (including single file with specific index)
    current_idx = args.start_idx
    end_idx = max(args.start_idx, args.end_idx)  # Ensure end_idx >= start_idx
    
    while current_idx <= end_idx:
        output_path = os.path.join(args.output_dir, f"realsense_hand_data_{current_idx:03d}.hdf5")
        
        print(f"\n{'='*60}")
        print(f"Recording file {current_idx} of {end_idx}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")
        
        # Run recording with multi-file mode
        recorder.run(output_path, args.generate_video, multi_file_mode=True, current_idx=current_idx, end_idx=end_idx)
        
        # Check if we should continue to next file or repeat
        if current_idx < end_idx:
            print(f"\nFile {current_idx} completed.")
            print("In the camera window, press:")
            print("  'n' - Next file")
            print("  'r' - Repeat current file") 
            print("  'q' - Quit")
            print("Waiting for your choice in the camera window...")
            
            # Wait for user input in the camera window
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    current_idx += 1
                    break
                elif key == ord('r'):
                    # Stay at current index to repeat
                    break
                elif key == ord('q'):
                    print("Quitting...")
                    # Clean up resources
                    if recorder.pipeline and recorder.pipeline_started:
                        try:
                            recorder.pipeline.stop()
                        except RuntimeError as e:
                            print(f"Warning: Failed to stop pipeline: {e}")
                    recorder.cleanup_audio()
                    cv2.destroyAllWindows()
                    return
        else:
            # Last file completed
            current_idx += 1
            break
        
    print(f"\nCompleted all recordings. Files saved to {args.output_dir}")
    
    # Final cleanup
    if recorder.pipeline and recorder.pipeline_started:
        try:
            recorder.pipeline.stop()
        except RuntimeError as e:
            print(f"Warning: Failed to stop pipeline: {e}")
    recorder.cleanup_audio()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
