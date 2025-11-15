#!/usr/bin/env python3
"""
RealSense to RoboMimic Data Processing Script
Processes RealSense recorded HDF5 data to extract hand poses and generate robomimic format

Features:
1. MediaPipe Hands detection for both hands
2. 3D coordinate estimation using depth information
3. SAM2 mask and line generation
4. Compatible output format with existing pipeline
5. Dynamic speed adjustment based on segment analysis
6. Optional audio augmentation via external noise WAV

Usage:
python realsense_to_robomimic.py --input_dir /path/to/hdf5/files --output /path/to/output.hdf5 --hand bimanual --enable_dynamic_speed [--augment noise.wav]

`--augment` and its alias `--augument` share the same behavior.
"""

import os
import h5py
import numpy as np
import cv2
import argparse
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import mediapipe as mp
import torch
import scipy.signal
from scipy.io import wavfile

# Import existing utilities
from mimicsonic.utils.mimicsonicUtils import (
    cam_frame_to_cam_pixels,
    WIDE_LENS_HAND_LEFT_K,
    interpolate_keys,
    interpolate_arr,
    REALSENSE_INTRINSICS
)
from mimicsonic.scripts.masking.utils import SAM, get_bounds, line_on_hand

from mimicsonic.scripts.aloha_process.aloha_to_robomimic import POINT_GAP
# Constants
HORIZON = 10
STEP = 3.0 * POINT_GAP / 2.0
DT = 0.02  # Time step duration in seconds

# Dynamic speed adjustment constants
DEFAULT_SEG_SPEED_RATIOS = [4, 4, 4]   # Robot/Human speed ratios for each seg
DEFAULT_ENABLE_DYNAMIC_SPEED = False  # Default: enabled for dynamic speed adjustment


def get_current_seg(frame_idx, mark_timesteps):
    """
    Determine which segment the current frame belongs to based on mark_timesteps
    
    Args:
        frame_idx: Current frame index
        mark_timesteps: Array of seg boundary timesteps (exclusive end points)
        
    Returns:
        seg_id: Segment ID (0, 1, 2, ...)
    """
    if mark_timesteps is None or len(mark_timesteps) == 0:
        return 0  # Default to seg 0 if no marks available
    
    for seg_id, end_frame in enumerate(mark_timesteps):
        if frame_idx < end_frame:
            return seg_id
    
    # If frame_idx is beyond all marks, return the last segment
    return len(mark_timesteps)


def get_dynamic_step(seg_id, base_step, seg_speed_ratios):
    """
    Calculate dynamic STEP value based on current segment
    
    Args:
        seg_id: Current segment ID
        base_step: Base STEP value (e.g., 3.0)
        seg_speed_ratios: Speed ratios for each segment
        
    Returns:
        dynamic_step: Adjusted STEP value
    """
    if seg_id < len(seg_speed_ratios):
        speed_ratio = seg_speed_ratios[seg_id]
        # STEP_new = STEP_original / speed_ratio
        # This makes human data slower to match robot speed
        return base_step / speed_ratio
    else:
        # Use default ratio for segments beyond the defined ratios
        return base_step / seg_speed_ratios[-1]


class HandPoseProcessor:
    """Main class for processing RealSense data with hand pose estimation"""
    
    def __init__(self, hand_type="bimanual", enable_dynamic_speed=DEFAULT_ENABLE_DYNAMIC_SPEED, seg_speed_ratios=None):
        """
        Initialize the hand pose processor
        
        Args:
            hand_type: "left", "right", or "bimanual"
            enable_dynamic_speed: Whether to enable dynamic speed adjustment
            seg_speed_ratios: Speed ratios for each segment (robot/human)
        """
        self.hand_type = hand_type
        self.enable_dynamic_speed = enable_dynamic_speed
        self.seg_speed_ratios = seg_speed_ratios or DEFAULT_SEG_SPEED_RATIOS
        self.seg_speed_ratios = [x / 4 for x in self.seg_speed_ratios]  # @gnq: divide by 4 to make human data slower to match robot speed
        # print(f"seg_speed_ratios: {self.seg_speed_ratios}")


        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize SAM for mask generation
        self.sam = SAM()
        
        # Camera intrinsics (will be loaded from HDF5)
        self.intrinsics = None
        
    def detect_hands_mediapipe(self, rgb_image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Detect hands using MediaPipe
        
        Args:
            rgb_image: RGB image (H, W, 3) - already in RGB format from HDF5
            
        Returns:
            Tuple of (left_hand_data, right_hand_data) or (None, None) if no hands detected
        """
        # Image is already in RGB format from HDF5, no conversion needed
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Remove this line
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                landmarks = np.array(landmarks)
                
                if handedness == "Left":
                    left_hand = {
                        'landmarks': landmarks,
                        'confidence': results.multi_handedness[idx].classification[0].score
                    }
                elif handedness == "Right":
                    right_hand = {
                        'landmarks': landmarks,
                        'confidence': results.multi_handedness[idx].classification[0].score
                    }
        
        return left_hand, right_hand
    
    def interpolate_depth(self, depth_image: np.ndarray, x_pixel: int, y_pixel: int, 
                         max_search_range: int = 50) -> float:
        """
        Interpolate depth value using nearby valid depth values from all four directions
        Uses progressive search strategy: 10 -> 20 -> 30 -> 50
        
        Args:
            depth_image: Depth image (H, W)
            x_pixel: X pixel coordinate
            y_pixel: Y pixel coordinate  
            max_search_range: Maximum search range for valid depth values (unused, kept for compatibility)
            
        Returns:
            Interpolated depth value in meters, or 0 if no valid depth found
        """
        height, width = depth_image.shape
        
        # Progressive search ranges: 10 -> 20 -> 30 -> 50
        search_ranges = [10, 20, 30, 50]
        
        # Search in all four directions: up, down, left, right
        directions = [
            (0, -1, "up"),      # (dx, dy, name)
            (0, 1, "down"),     # (dx, dy, name)
            (-1, 0, "left"),    # (dx, dy, name)
            (1, 0, "right")     # (dx, dy, name)
        ]
        
        # Try each search range progressively
        for search_range in search_ranges:
            valid_depths = []
            
            for dx, dy, direction_name in directions:
                for distance in range(1, search_range + 1):
                    check_x = x_pixel + dx * distance
                    check_y = y_pixel + dy * distance
                    
                    # Check bounds
                    if 0 <= check_x < width and 0 <= check_y < height:
                        depth_value = depth_image[check_y, check_x]
                        if depth_value > 0:
                            # Store depth value with distance for weighting
                            valid_depths.append((depth_value, distance, direction_name))
                            break  # Found valid depth in this direction, move to next direction
            
            # If we found valid depths at this search range, use them
            if valid_depths:
                # Interpolation strategies based on number of valid depths found
                if len(valid_depths) == 1:
                    # Only one valid depth found
                    return valid_depths[0][0] / 1000.0
                
                elif len(valid_depths) == 2:
                    # Two valid depths found - use simple average
                    avg_depth = sum(depth[0] for depth in valid_depths) / len(valid_depths)
                    return avg_depth / 1000.0
                
                else:
                    # Multiple valid depths found - use distance-weighted average
                    # Closer depths get higher weight
                    total_weight = 0.0
                    weighted_sum = 0.0
                    
                    for depth_value, distance, _ in valid_depths:
                        # Weight inversely proportional to distance (closer = higher weight)
                        weight = 1.0 / distance
                        weighted_sum += depth_value * weight
                        total_weight += weight
                    
                    avg_depth = weighted_sum / total_weight
                    return avg_depth / 1000.0
        
        # No valid depth found in any search range
        return 0.0

    def convert_to_3d_coordinates(self, hand_landmarks: np.ndarray, depth_image: np.ndarray, 
                                 intrinsics: np.ndarray) -> np.ndarray:
        """
        Convert 2D hand landmarks to 3D coordinates using depth information
        
        Args:
            hand_landmarks: Hand landmarks from MediaPipe (21, 3) - normalized coordinates
            depth_image: Depth image (H, W)
            intrinsics: Camera intrinsics matrix (3, 4)
            
        Returns:
            3D coordinates (21, 3) in camera frame
        """
        height, width = depth_image.shape
        landmarks_3d = []
        
        for landmark in hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            x_pixel = int(landmark[0] * width)
            y_pixel = int(landmark[1] * height)
            
            # Clamp to image bounds
            x_pixel = max(0, min(width - 1, x_pixel))
            y_pixel = max(0, min(height - 1, y_pixel))
            
            # Get depth value
            depth = depth_image[y_pixel, x_pixel]
            # print(f"depth: {depth}")
            # exit()
            
            if depth > 0:  # Valid depth
                # Convert depth from millimeters to meters
                depth_meters = depth / 1000.0
            else:
                # Try to interpolate depth from nearby pixels
                depth_meters = self.interpolate_depth(depth_image, x_pixel, y_pixel)
                
                if depth_meters == 0.0:
                    # Still no valid depth after interpolation - raise error
                    raise ValueError(f"No valid depth found at pixel ({x_pixel}, {y_pixel}) "
                                   f"even after interpolation. This indicates a serious problem "
                                   f"with depth data in this region.")
            
            # Convert to 3D coordinates using camera intrinsics
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            # print("landmark:", landmark)
            # print("height, width:", height, width)
            # print(f"x_pixel: {x_pixel}, y_pixel: {y_pixel}, depth_meters: {depth_meters}")
            # print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
            # exit()

            x_3d = (x_pixel - cx) * depth_meters / fx
            y_3d = (y_pixel - cy) * depth_meters / fy
            z_3d = depth_meters
            
            landmarks_3d.append([x_3d, y_3d, z_3d])
            # print(f"x_3d: {x_3d}, y_3d: {y_3d}, z_3d: {z_3d}")
            # exit()
        
        return np.array(landmarks_3d)
    
    def extract_palm_center(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Extract palm center from hand landmarks
        
        Args:
            landmarks_3d: 3D landmarks (21, 3)
            
        Returns:
            Palm center coordinates (3,)
        """
        # Use wrist (landmark 0) as palm center
        return landmarks_3d[0]
    
    

    def generate_masks_and_lines(self, rgb_images: np.ndarray, hand_coords: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks and lines using SAM2
        
        Args:
            rgb_images: RGB images (N, H, W, 3)
            hand_coords: List of hand coordinates for each frame
            
        Returns:
            Tuple of (overlayed_images, masked_images, raw_masks)
        """
        # Convert hand coordinates to pixel coordinates for SAM
        ee_poses = []
        for coords in hand_coords:
            if self.hand_type == "bimanual":
                # Combine left and right hand coordinates
                if len(coords) == 2:  # [left_hand, right_hand]
                    right_hand = coords[0] if coords[0] is not None else np.zeros(3)
                    left_hand = coords[1] if coords[1] is not None else np.zeros(3)
                    ee_poses.append(np.concatenate([left_hand, right_hand]))
                else:
                    ee_poses.append(np.zeros(6))
            else:
                # Single hand
                if coords is not None:
                    ee_poses.append(coords)
                else:
                    ee_poses.append(np.zeros(3))
        
        ee_poses = np.array(ee_poses)
        
        print(f"About to call sam.get_hand_mask_line_batched with ee_poses shape: {ee_poses.shape}")
        # Use SAM to generate masks and lines
        overlayed_imgs, masked_imgs, raw_masks = self.sam.get_hand_mask_line_batched(
            rgb_images, ee_poses, REALSENSE_INTRINSICS, debug=False
        )
        print("sam.get_hand_mask_line_batched completed")

        # import matplotlib.pyplot as plt
        # idx = 4  # 你可以改成 5、10 等看看不同帧
        # original_img = rgb_images[idx]
        # line_img = masked_imgs[idx]
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_img)
        # plt.title("Original Image")
        # plt.axis("off")
        # plt.subplot(1, 2, 2)
        # plt.imshow(line_img)
        # plt.title("Line Image")
        # plt.axis("off")
        # plt.tight_layout()
        # plt.show()

        # show_images_auto(raw_masks, masked_imgs,overlayed_imgs)  # @gnq

        return overlayed_imgs, masked_imgs, raw_masks
    
    def process_single_hdf5(self, hdf5_path: str, enable_audio=False, audio_length=2.0, target_sampling_rate=16000, noise_data: Optional[np.ndarray] = None, noise_volume: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Process a single HDF5 file
        
        Args:
            hdf5_path: Path to HDF5 file
            enable_audio: Whether to process audio data
            audio_length: Audio length in seconds
            target_sampling_rate: Target audio sampling rate
            noise_data: Optional augmentation noise waveform already normalized/resampled
            noise_volume: Scaling factor applied to the noise before mixing
            
        Returns:
            Tuple of (actions, images, ee_poses, overlayed_imgs, masked_imgs, raw_masks, audio_data)
        """
        print(f"Processing {hdf5_path}...")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Check if this is raw RealSense data or processed data
            if 'rgb' in f and 'depth' in f:
                # Raw RealSense data format
                print("Detected raw RealSense data format")
                rgb_images = f['rgb'][:]
                depth_images = f['depth'][:]
                timestamps = f['timestamps'][:]
                self.intrinsics = REALSENSE_INTRINSICS
                
                # Load mark_timesteps for dynamic speed adjustment
                mark_timesteps = None
                if 'mark_timesteps' in f:
                    mark_timesteps = f['mark_timesteps'][:]
                    print(f"Loaded mark_timesteps: {mark_timesteps}")
                
                # Load audio data if available and enabled
                audio_data_raw = None
                original_sampling_rate = 48000  # Default
                fps = 30  # Default
                
                if enable_audio and 'audio' in f:
                    audio_data_raw = f['audio'][:]  # Shape: (num_timesteps, 2048)
                    original_sampling_rate = f.attrs.get('audio_sampling_rate', 48000)
                    fps = f.attrs.get('fps', 30)
                    print(f"Loaded audio data: shape {audio_data_raw.shape}, original rate: {original_sampling_rate}Hz")
                elif enable_audio:
                    print("Warning: Audio processing enabled but no audio data found in HDF5 file")

                # import matplotlib.pyplot as plt
                # idx = 0
                # rgb_image = rgb_images[idx]
                # plt.figure(figsize=(10, 5))
                # plt.subplot(1, 2, 1)
                # plt.imshow(rgb_image)
                # plt.title("Original Image")
                # plt.axis("off")
                # plt.subplot(1, 2, 2)
                # plt.imshow(rgb_image)
                # plt.title("Line Image")
                # plt.axis("off")
                # plt.tight_layout()
                # plt.show()

                # Load camera intrinsics if available
                # if 'color_intrinsics' in f.attrs:
                #     color_intrinsics = json.loads(f.attrs['color_intrinsics'])
                #     self.intrinsics = np.array([
                #         [color_intrinsics['fx'], 0, color_intrinsics['ppx'], 0],
                #         [0, color_intrinsics['fy'], color_intrinsics['ppy'], 0],
                #         [0, 0, 1, 0]
                #     ])
                # else:
                #     # Use default RealSense intrinsics
                #     self.intrinsics = REALSENSE_INTRINSICS
                
            elif 'data' in f:
                # Processed data format - skip this file
                print(f"Skipping {hdf5_path} - already processed data")
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None
            else:
                print(f"Unknown data format in {hdf5_path}")
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None
        
        print(f"Loaded {len(rgb_images)} frames")
        
        # Process frames following aria_to_robomimic.py logic
        frame_length = len(rgb_images) - 1
        print(f"Processing {frame_length + 1} frames")
        
        actions = []
        front_img_1 = []
        ee_pose = []
        
        for t in tqdm(range(frame_length + 1), desc="Processing frames"):
            # Determine current segment and dynamic STEP
            current_seg = get_current_seg(t, mark_timesteps)
            if self.enable_dynamic_speed:
                dynamic_step = get_dynamic_step(current_seg, STEP, self.seg_speed_ratios)
                # print(f"dynamic_step: {dynamic_step}")
                # if t % 100 == 0:  # Log every 100 frames
                #     print(f"Frame {t}: seg={current_seg}, dynamic_step={dynamic_step:.3f}")
            else:
                dynamic_step = STEP

            # Check if we have enough future frames (following aria logic)
            if t + HORIZON * dynamic_step < frame_length + 1:
                if (t % 1000) == 0:
                    print(f"{t} frames ingested")
                
                # Get current frame data
                rgb_image = rgb_images[t]
                depth_image = depth_images[t]
                
                # Detect hands for current frame
                left_hand, right_hand = self.detect_hands_mediapipe(rgb_image)
               
                # Extract current frame hand coordinates
                current_hand_coords = None
                current_ee_pose = None
                
                if self.hand_type == "left":
                    if left_hand is not None:
                        landmarks_3d = self.convert_to_3d_coordinates(
                            left_hand['landmarks'], depth_image, self.intrinsics
                        )
                        current_hand_coords = self.extract_palm_center(landmarks_3d)
                        current_ee_pose = current_hand_coords
                    else:
                        continue  # Skip if no hand detected
                        
                elif self.hand_type == "right":
                    if right_hand is not None:
                        landmarks_3d = self.convert_to_3d_coordinates(
                            right_hand['landmarks'], depth_image, self.intrinsics
                        )
                        current_hand_coords = self.extract_palm_center(landmarks_3d)
                        current_ee_pose = current_hand_coords
                    else:
                        continue  # Skip if no hand detected
                        
                elif self.hand_type == "bimanual":
                    left_coords = None
                    right_coords = None
                    
                    if left_hand is not None:
                        landmarks_3d = self.convert_to_3d_coordinates(
                            left_hand['landmarks'], depth_image, self.intrinsics
                        )
                        left_coords = self.extract_palm_center(landmarks_3d)
                        # print(f"left_hand['landmarks']: {left_hand['landmarks'][0]}")
                        # print(f"landmarks_3d: {landmarks_3d[0]}")
                        # print(f"left_coords: {left_coords[0]}")
                        # exit()
                    
                    if right_hand is not None:
                        landmarks_3d = self.convert_to_3d_coordinates(
                            right_hand['landmarks'], depth_image, self.intrinsics
                        )
                        right_coords = self.extract_palm_center(landmarks_3d)
                    
                    if left_coords is not None or right_coords is not None:
                        # Store as [left_coords, right_coords] format for mask generation
                        current_hand_coords = [left_coords, right_coords]
                        # Create flattened ee_pose for actions
                        if left_coords is not None and right_coords is not None:
                            current_ee_pose = np.concatenate([left_coords, right_coords])
                            # print(f"current_ee_pose: {current_ee_pose}")
                            # exit()
                        elif left_coords is not None:
                            current_ee_pose = np.concatenate([left_coords, np.zeros(3)])
                        elif right_coords is not None:
                            current_ee_pose = np.concatenate([np.zeros(3), right_coords])
                    else:
                        continue  # Skip if no hands detected
                
                # Generate future action sequence (following aria logic)
                actions_t = []
                for offset in range(HORIZON):
                    future_frame_idx = int(t + offset * dynamic_step)
                    if future_frame_idx < len(rgb_images):
                        future_rgb = rgb_images[future_frame_idx]
                        future_depth = depth_images[future_frame_idx]
                        
                        # Detect hands for future frame
                        future_left_hand, future_right_hand = self.detect_hands_mediapipe(future_rgb)
                        
                        if self.hand_type == "left":
                            if future_left_hand is not None:
                                landmarks_3d = self.convert_to_3d_coordinates(
                                    future_left_hand['landmarks'], future_depth, self.intrinsics
                                )
                                future_coords = self.extract_palm_center(landmarks_3d)
                            else:
                                future_coords = np.zeros(3)
                            actions_t.append(future_coords)
                            
                        elif self.hand_type == "right":
                            if future_right_hand is not None:
                                landmarks_3d = self.convert_to_3d_coordinates(
                                    future_right_hand['landmarks'], future_depth, self.intrinsics
                                )
                                future_coords = self.extract_palm_center(landmarks_3d)
                            else:
                                future_coords = np.zeros(3)
                            actions_t.append(future_coords)
                            
                        elif self.hand_type == "bimanual":
                            left_coords = np.zeros(3)
                            right_coords = np.zeros(3)
                            
                            if future_left_hand is not None:
                                landmarks_3d = self.convert_to_3d_coordinates(
                                    future_left_hand['landmarks'], future_depth, self.intrinsics
                                )
                                left_coords = self.extract_palm_center(landmarks_3d)
                            
                            if future_right_hand is not None:
                                landmarks_3d = self.convert_to_3d_coordinates(
                                    future_right_hand['landmarks'], future_depth, self.intrinsics
                                )
                                right_coords = self.extract_palm_center(landmarks_3d)
                            
                            actions_t.append(np.concatenate([left_coords, right_coords]))
                    else:
                        # Use last available frame if not enough future frames
                        if self.hand_type == "bimanual":
                            actions_t.append(np.zeros(6))
                        else:
                            actions_t.append(np.zeros(3))
                
                # Only add if we have valid current frame data
                if current_hand_coords is not None and current_ee_pose is not None:
                    actions.append(np.array(actions_t))
                    front_img_1.append(rgb_image)
                    ee_pose.append(current_ee_pose)
        
        if len(actions) == 0:
            print("Warning: No valid sequences generated")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None
        
        # Convert to numpy arrays
        actions = np.array(actions)
        front_img_1 = np.array(front_img_1)
        ee_pose = np.array(ee_pose)
        
        print(f"Generated {len(actions)} valid sequences")
        
        # Generate masks and lines
        print("Generating masks and lines")
        # Create hand_coords list in the format expected by generate_masks_and_lines
        hand_coords_for_masks = []
        for i in range(len(ee_pose)):
            if self.hand_type == "bimanual":
                # Convert 6D ee_pose back to [left_coords, right_coords] format
                left_coords = ee_pose[i][:3] if np.any(ee_pose[i][:3] != 0) else None
                right_coords = ee_pose[i][3:6] if np.any(ee_pose[i][3:6] != 0) else None
                hand_coords_for_masks.append([left_coords, right_coords])
            else:
                # Single hand
                hand_coords_for_masks.append(ee_pose[i])
        
        overlayed_imgs, masked_imgs, raw_masks = self.generate_masks_and_lines(
            front_img_1, hand_coords_for_masks
        )
        
        # Process audio data if available
        processed_audio = None
        if enable_audio and audio_data_raw is not None:
            print("Processing audio data...")
            # Extract and concatenate audio data from 2D format
            # Format: (num_timesteps, max_samples), last column stores actual length
            concatenated_audio = []
            for t in range(audio_data_raw.shape[0]):
                # Get actual length (stored in last element)
                actual_length = int(audio_data_raw[t, -1])
                if actual_length > 0:
                    # Extract valid audio data (excluding the length element)
                    timestep_audio = audio_data_raw[t, :actual_length]
                    concatenated_audio.append(timestep_audio)
            
            if concatenated_audio:
                # Concatenate all timesteps into 1D array
                audio_data_1d = np.concatenate(concatenated_audio)
                print(f"Extracted {len(audio_data_1d)} audio samples from {len(concatenated_audio)} timesteps")
                
                processed_audio = process_audio_data(
                    audio_data_1d, 
                    len(front_img_1), 
                    audio_length=audio_length,
                    target_sampling_rate=target_sampling_rate,
                    original_sampling_rate=original_sampling_rate,
                    fps=fps,
                    noise_data=noise_data,
                    noise_volume=noise_volume
                )
            else:
                print("Warning: No valid audio data extracted from HDF5")
        
        return actions, front_img_1, ee_pose, overlayed_imgs, masked_imgs, raw_masks, processed_audio


def load_noise_audio(noise_path: str, target_sampling_rate: int) -> Optional[np.ndarray]:
    """Load and normalize noise audio for augmentation."""
    if not os.path.exists(noise_path):
        print(f"❌ Noise audio file not found: {noise_path}")
        return None

    try:
        source_rate, data = wavfile.read(noise_path)
        print(f"Loaded noise audio '{noise_path}' ({len(data)} samples @ {source_rate}Hz)")
    except Exception as exc:
        print(f"❌ Failed to read noise audio '{noise_path}': {exc}")
        return None

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    orig_dtype = data.dtype
    data = data.astype(np.float32)

    if np.issubdtype(orig_dtype, np.integer):
        dtype_info = np.iinfo(orig_dtype)
        max_abs = max(abs(dtype_info.min), abs(dtype_info.max))
        if max_abs > 0:
            data /= max_abs
    else:
        max_abs = np.max(np.abs(data)) if len(data) > 0 else 1.0
        if max_abs > 1.0:
            data /= max_abs

    if source_rate != target_sampling_rate and len(data) > 0:
        resample_ratio = target_sampling_rate / source_rate
        new_length = max(1, int(len(data) * resample_ratio))
        data = scipy.signal.resample(data, new_length).astype(np.float32)

    if len(data) == 0:
        print(f"❌ Noise audio '{noise_path}' is empty after processing.")
        return None

    print(f"✅ Loaded noise audio '{noise_path}' ({len(data)} samples @ {target_sampling_rate}Hz)")
    return data


def process_audio_data(audio_data, num_frames, audio_length=2.0, target_sampling_rate=16000, original_sampling_rate=48000, fps=30, noise_data: Optional[np.ndarray] = None, noise_volume: float = 0.0):
    """
    Process audio data by concatenating previous timesteps and aligning with video frames.
    This follows the same logic as aloha_to_robomimic.py - collecting previous 2 seconds of audio.
    
    Args:
        audio_data: Raw audio data (1D array)
        num_frames: Number of video frames (for alignment)
        audio_length: Desired audio length in seconds
        target_sampling_rate: Target sampling rate in Hz
        original_sampling_rate: Original sampling rate in Hz
        fps: Video frame rate
        noise_data: Optional normalized noise waveform for augmentation
        noise_volume: Scaling factor applied to the noise when mixing
    
    Returns:
        processed_audio: Processed audio data aligned with video frames
    """
    print(f"Processing audio: {len(audio_data)} samples, original rate: {original_sampling_rate}Hz")
    
    # Calculate audio samples per video frame
    samples_per_frame = original_sampling_rate / fps
    samples_per_frame_int = max(1, int(np.ceil(samples_per_frame)))
    
    # Calculate target audio length in samples
    max_audio_length = int(target_sampling_rate * audio_length)
    
    # Calculate how many frames correspond to the audio length
    frames_per_audio_length = max(1, int(audio_length * fps))
    
    print(f"Audio processing parameters:")
    print(f"  samples_per_frame: {samples_per_frame}")
    print(f"  frames_per_audio_length: {frames_per_audio_length}")
    print(f"  max_audio_length: {max_audio_length}")
    
    def sample_noise_segment(noise_arr: Optional[np.ndarray], length: int) -> Optional[np.ndarray]:
        if noise_arr is None or len(noise_arr) == 0 or length <= 0:
            return None
        start = np.random.randint(0, len(noise_arr))
        if start + length <= len(noise_arr):
            return noise_arr[start:start + length].copy()
        segment = []
        needed = length
        cursor = start
        while needed > 0:
            end = min(len(noise_arr), cursor + needed)
            segment.append(noise_arr[cursor:end])
            needed -= (end - cursor)
            cursor = 0
        return np.concatenate(segment) if segment else None
    
    # Process each frame
    processed_audio_frames = []
    
    for frame_idx in range(num_frames):
        # Calculate the start frame for the audio window (previous 2 seconds)
        start_frame = max(0, frame_idx - (frames_per_audio_length - 1))
        
        # Collect audio data from start_frame to current frame
        selected_audio = []
        for f in range(start_frame, frame_idx + 1):
            # Calculate audio sample range for this frame
            start_sample = int(f * samples_per_frame)
            end_sample = int((f + 1) * samples_per_frame)
            
            # Extract audio for this frame
            if end_sample <= len(audio_data):
                frame_audio = audio_data[start_sample:end_sample]
            else:
                # Handle case where we're beyond the audio data
                frame_audio = np.zeros(samples_per_frame_int, dtype=np.int16)
            
            selected_audio.append(frame_audio)
        
        # Handle missing frames by padding with zeros at the beginning
        missing_frames = frames_per_audio_length - len(selected_audio)
        if missing_frames > 0:
            zero_padding = [np.zeros(samples_per_frame_int, dtype=np.int16)] * missing_frames
            selected_audio = zero_padding + selected_audio
        
        # Concatenate all audio data
        concatenated_audio = np.concatenate(selected_audio) if selected_audio else np.zeros(0, dtype=np.int16)
        pre_resample_pad_samples = max(0, missing_frames) * samples_per_frame_int
        resample_ratio = 1.0
        
        # Resample if needed
        if original_sampling_rate != target_sampling_rate and len(concatenated_audio) > 0:
            resample_ratio = target_sampling_rate / original_sampling_rate
            new_length = max(1, int(len(concatenated_audio) * resample_ratio))
            concatenated_audio = scipy.signal.resample(concatenated_audio, new_length)
        else:
            concatenated_audio = concatenated_audio.astype(np.float32)
        
        protected_samples = int(pre_resample_pad_samples * resample_ratio)
        
        # Adjust length to target
        current_len = len(concatenated_audio)
        if current_len > max_audio_length:
            # Truncate from the end (keep most recent audio)
            trimmed = current_len - max_audio_length
            protected_samples = max(protected_samples - trimmed, 0)
            concatenated_audio = concatenated_audio[-max_audio_length:]
        elif current_len < max_audio_length:
            # Pad with zeros at the beginning
            padding = np.zeros((max_audio_length - current_len,), dtype=np.float32)
            concatenated_audio = np.concatenate([padding, concatenated_audio.astype(np.float32)])
            protected_samples = min(max_audio_length, max_audio_length - current_len + protected_samples)
        else:
            concatenated_audio = concatenated_audio.astype(np.float32)
            protected_samples = min(protected_samples, current_len)
        
        protected_samples = min(protected_samples, len(concatenated_audio))
        
        # Normalize audio to [-1, 1] range
        concatenated_audio = concatenated_audio / 32768.0

        if (
            noise_data is not None
            and noise_volume > 0.0
            and len(concatenated_audio) > 0
            and protected_samples < len(concatenated_audio)
        ):
            actual_len = len(concatenated_audio) - protected_samples
            noise_segment = sample_noise_segment(noise_data, actual_len)
            if noise_segment is not None and len(noise_segment) == actual_len:
                concatenated_audio = concatenated_audio.copy()
                concatenated_audio[protected_samples:] = np.clip(
                    concatenated_audio[protected_samples:] + noise_volume * noise_segment,
                    -1.0,
                    1.0,
                )

        processed_audio_frames.append(concatenated_audio)
        
        # Debug info for first few frames
        if frame_idx < 5:
            print(f"Frame {frame_idx}: window [{start_frame}, {frame_idx}], audio length: {len(concatenated_audio)}")
    
    processed_audio = np.array(processed_audio_frames, dtype=np.float32)
    print(f"Processed audio shape: {processed_audio.shape}")
    return processed_audio


def transform_actions(actions):
    """Transform coordinates for actions"""
    print("Transforming coordinates for actions")
    
    if len(actions.shape) == 3:  # (N, horizon, ac_dim)
        if actions.shape[2] == 3:
            actions[:, :, 0] *= -1  # Multiply x by -1
            actions[:, :, 1] *= -1  # Multiply y by -1
        elif actions.shape[2] == 6:
            actions[:, :, 0] *= -1  # Multiply x by -1 for first set
            actions[:, :, 1] *= -1  # Multiply y by -1 for first set
            actions[:, :, 3] *= -1  # Multiply x by -1 for second set
            actions[:, :, 4] *= -1  # Multiply y by -1 for second set
    
    return actions


def transform_ee_pose(ee_pose):
    """Transform coordinates for ee_pose"""
    if len(ee_pose.shape) == 2:  # (N, ac_dim) - current frame positions
        if ee_pose.shape[1] == 3:
            ee_pose[:, 0] *= -1  # Multiply x by -1
            ee_pose[:, 1] *= -1  # Multiply y by -1
        elif ee_pose.shape[1] == 6:
            ee_pose[:, 0] *= -1  # Multiply x by -1 for first set
            ee_pose[:, 1] *= -1  # Multiply y by -1 for first set
            ee_pose[:, 3] *= -1  # Multiply x by -1 for second set
            ee_pose[:, 4] *= -1  # Multiply y by -1 for second set
    elif len(ee_pose.shape) == 3:  # (N, horizon, ac_dim) - legacy format
        if ee_pose.shape[2] == 3:
            ee_pose[:, :, 0] *= -1  # Multiply x by -1
            ee_pose[:, :, 1] *= -1  # Multiply y by -1
        elif ee_pose.shape[2] == 6:
            ee_pose[:, :, 0] *= -1  # Multiply x by -1 for first set
            ee_pose[:, :, 1] *= -1  # Multiply y by -1 for first set
            ee_pose[:, :, 3] *= -1  # Multiply x by -1 for second set
            ee_pose[:, :, 4] *= -1  # Multiply y by -1 for second set

    return ee_pose


def split_train_val_from_hdf5(hdf5_path, val_ratio):
    with h5py.File(hdf5_path, "a") as file:
        demo_keys = [key for key in file["data"].keys() if "demo" in key]
        num_demos = len(demo_keys)
        num_val = int(np.ceil(num_demos * val_ratio))

        indices = np.arange(num_demos)
        np.random.shuffle(indices)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_mask = [f"demo_{i}" for i in train_indices]
        val_mask = [f"demo_{i}" for i in val_indices]

        file.create_dataset("mask/train", data=np.array(train_mask, dtype="S"))
        file.create_dataset("mask/valid", data=np.array(val_mask, dtype="S"))

def main(args):
    """Main processing function"""
    # Find all HDF5 files in input directory
    hdf5_files = [f for f in os.listdir(args.input_dir) if f.endswith('.hdf5')]
    hdf5_files.sort()
    
    if not hdf5_files:
        print(f"No HDF5 files found in {args.input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files: {hdf5_files}")
    
    # Initialize processor with dynamic speed adjustment
    processor = HandPoseProcessor(
        hand_type=args.hand,
        enable_dynamic_speed=args.enable_dynamic_speed,
        seg_speed_ratios=args.seg_speed_ratios
    )
    
    noise_data = None
    noise_volume = max(args.augment_volume, 0.0)
    if args.augment:
        noise_data = load_noise_audio(args.augment, args.target_sampling_rate)
        if noise_data is None:
            print("⚠️  Noise augmentation disabled due to load failure.")
            noise_volume = 0.0
    else:
        noise_volume = 0.0

    # Process all files
    all_actions = []
    all_images = []
    all_ee_poses = []
    all_overlayed_imgs = []
    all_masked_imgs = []
    all_raw_masks = []
    all_audio_data = []
    
    for hdf5_file in hdf5_files:
        hdf5_path = os.path.join(args.input_dir, hdf5_file)
        
        actions, images, ee_poses, overlayed_imgs, masked_imgs, raw_masks, audio_data = processor.process_single_hdf5(
            hdf5_path, 
            enable_audio=args.enable_audio,
            audio_length=args.audio_length,
            target_sampling_rate=args.target_sampling_rate,
            noise_data=noise_data,
            noise_volume=noise_volume
        )
        
        if len(actions) > 0:
            all_actions.append(actions)
            all_images.append(images)
            all_ee_poses.append(ee_poses)
            all_overlayed_imgs.append(overlayed_imgs)
            all_masked_imgs.append(masked_imgs)
            all_raw_masks.append(raw_masks)
            if audio_data is not None:
                all_audio_data.append(audio_data)
    
    if not all_actions:
        print("No valid data processed")
        return
    
    # Concatenate all data
    all_actions = np.concatenate(all_actions, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_ee_poses = np.concatenate(all_ee_poses, axis=0)
    all_overlayed_imgs = np.concatenate(all_overlayed_imgs, axis=0)
    all_masked_imgs = np.concatenate(all_masked_imgs, axis=0)
    all_raw_masks = np.concatenate(all_raw_masks, axis=0)
    
    # Concatenate audio data if available
    if all_audio_data:
        all_audio_data = np.concatenate(all_audio_data, axis=0)
        print(f"Total audio data: {all_audio_data.shape}")
    else:
        all_audio_data = None
    
    print(f"Total processed data: {len(all_actions)} sequences")
    
    # Transform coordinates
    all_actions = transform_actions(all_actions)
    all_ee_poses = transform_ee_pose(all_ee_poses)
    
    # Save to HDF5
    print(f"Saving to {args.output}...")

    with h5py.File(args.output, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps({})

        # Determine action dimension
        if args.hand in ("left", "right"):
            ac_dim = 3
        else:  # bimanual
            ac_dim = 6

        demo_index = 0
        chunk_size = 300

        # 安全检查：确保所有数组长度一致
        total = min(len(all_actions), len(all_images), len(all_ee_poses),
                    len(all_overlayed_imgs), len(all_masked_imgs), len(all_raw_masks))
        
        # 如果有音频数据，也要检查音频数据的长度
        if all_audio_data is not None:
            total = min(total, len(all_audio_data))

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            cur_len = end - start

            group = data.create_group(f"demo_{demo_index}")

            # ---- Actions ----
            ac_reshape = all_actions[start:end]                       # (cur_len, horizon, ac_dim)
            group.create_dataset("actions_xyz", data=ac_reshape)
            ac_reshape_interp = interpolate_arr(ac_reshape, 100)       # 你的插值函数
            group.create_dataset("actions_xyz_act", data=ac_reshape_interp)
            group.attrs["num_samples"] = ac_reshape.shape[0]

            # ---- Images (RGB) ----
            rgb_images = all_images[start:end]                        # 期望形状 (cur_len, H, W, 3)
            # 若你的源数据是 BGR，则转 RGB：[..., ::-1]
            # if rgb_images.ndim == 4 and rgb_images.shape[-1] == 3:
            #     # 如果已经是 RGB，这一步不会坏；如果是 BGR，这一步会转成 RGB
            #     rgb_images_rgb = rgb_images[..., ::-1]
            # else:
            #     raise ValueError(f"Unexpected rgb_images shape: {rgb_images.shape}")

            rgb_images_rgb = rgb_images # @gnq

            # ---- Masks & Overlays (也保证是 RGB 排列) ----
            masked_imgs = all_masked_imgs[start:end]

            # if masked_imgs.ndim == 4 and masked_imgs.shape[-1] == 3:
            #     masked_imgs_rgb = masked_imgs[..., ::-1]
            # else:
            #     raise ValueError(f"Unexpected masked_imgs shape: {masked_imgs.shape}")

            masked_imgs_rgb = masked_imgs # @gnq

            # Save masked image to front_img_1 (main image for training)
            group.create_dataset("obs/front_img_1", data=masked_imgs_rgb)   # modified by @gnq
            
            # Save original image to front_img_1_original
            group.create_dataset("obs/front_img_1_original", data=rgb_images_rgb)  # modified by @gnq

            # ---- EE Pose ----
            group.create_dataset("obs/ee_pose", data=all_ee_poses[start:end])

            # Keep front_img_1_masked for backward compatibility (same as front_img_1 now)
            group.create_dataset("obs/front_img_1_masked", data=masked_imgs_rgb)

            overlayed_imgs = all_overlayed_imgs[start:end]
            # if overlayed_imgs.ndim == 4 and overlayed_imgs.shape[-1] == 3:
            #     overlayed_imgs_rgb = overlayed_imgs[..., ::-1]
            # else:
            #     raise ValueError(f"Unexpected overlayed_imgs shape: {overlayed_imgs.shape}")
            overlayed_imgs_rgb = overlayed_imgs # @gnq
            group.create_dataset("obs/front_img_1_line", data=overlayed_imgs_rgb)

            group.create_dataset("obs/front_img_1_mask", data=all_raw_masks[start:end])

            # ---- Audio Data ----
            if all_audio_data is not None:
                group.create_dataset("obs/audio", data=all_audio_data[start:end])
                # Store audio metadata as attributes
                group.attrs["audio_sampling_rate"] = args.target_sampling_rate
                group.attrs["audio_length"] = args.audio_length
                print(f"Added audio data to demo {demo_index}: shape {all_audio_data[start:end].shape}")

            demo_index += 1

    print(f"Saved {demo_index} demos to {args.output}")
    
    # Split into train/val
    if args.split:
        split_train_val_from_hdf5(args.output, args.val_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense to RoboMimic Data Processing")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing HDF5 files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file path"
    )
    
    parser.add_argument(
        "--hand",
        type=str,
        choices=["left", "right", "bimanual"],
        default="bimanual",
        help="Hand type to process"
    )
    
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split dataset into train/validation sets"
    )
    
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode"
    )
    
    # Audio processing arguments
    parser.add_argument(
        "--enable_audio",
        action="store_true",
        help="Enable audio processing"
    )
    parser.add_argument(
        "--audio_length",
        type=float,
        default=2.0,
        help="Audio length in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--target_sampling_rate",
        type=int,
        default=16000,
        help="Target audio sampling rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--augment",
        "--augument",
        dest="augment",
        type=str,
        help="Path to noise audio (WAV) for augmentation; omitted means no noise"
    )
    parser.add_argument(
        "--augment_volume",
        type=float,
        default=0.7,
        help="Scaling factor applied to augmentation noise (default: 0.7)"
    )
    
    # Dynamic speed adjustment arguments
    parser.add_argument(
        "--enable_dynamic_speed",
        action="store_true",
        help="Enable dynamic speed adjustment based on segment analysis"
    )
    parser.add_argument(
        "--seg_speed_ratios",
        type=float,
        nargs='+',
        default=DEFAULT_SEG_SPEED_RATIOS,
        help=f"Speed ratios for each segment (default: {DEFAULT_SEG_SPEED_RATIOS})"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        exit(1)
    
    # Validate dynamic speed adjustment parameters
    if args.enable_dynamic_speed:
        if len(args.seg_speed_ratios) == 0:
            print("Error: seg_speed_ratios cannot be empty when dynamic speed is enabled")
            exit(1)
        if any(ratio <= 0 for ratio in args.seg_speed_ratios):
            print("Error: All speed ratios must be positive")
            exit(1)
        print(f"Dynamic speed adjustment enabled with ratios: {args.seg_speed_ratios}")
    else:
        print("Dynamic speed adjustment disabled (using original STEP=3.0)")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(args)