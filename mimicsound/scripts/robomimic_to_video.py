#!/usr/bin/env python3
"""
Convert RoboMimic HDF5 data to video

将robomimic格式的HDF5文件中的图像数据转换为视频文件
支持多种图像类型：front_img_1, front_img_1_masked, front_img_1_line, front_img_1_mask

Usage:
python scripts/robomimic_to_video.py --input /path/to/robomimic.hdf5 --output /path/to/output.mp4 --image_type front_img_1_masked --demo_id 0
"""

import os
import h5py
import numpy as np
import cv2
import argparse
from typing import Optional

def load_robomimic_data(hdf5_path: str, demo_id: int = 0, image_type: str = "front_img_1_masked") -> Optional[np.ndarray]:
    """
    Load image data from RoboMimic HDF5 file
    
    Args:
        hdf5_path: Path to RoboMimic HDF5 file
        demo_id: Demo ID to extract (default: 0)
        image_type: Type of image to extract (front_img_1, front_img_1_masked, front_img_1_line, front_img_1_mask)
        
    Returns:
        Image data as numpy array (N, H, W, 3) or None if not found
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check if data group exists
            if 'data' not in f:
                print(f"❌ No 'data' group found in {hdf5_path}")
                return None
            
            # Check if demo exists
            demo_key = f"demo_{demo_id}"
            if demo_key not in f['data']:
                available_demos = [key for key in f['data'].keys() if key.startswith('demo_')]
                print(f"❌ Demo {demo_id} not found. Available demos: {available_demos}")
                return None
            
            demo_group = f['data'][demo_key]
            
            # Check if image type exists (might be in obs group)
            if image_type in demo_group:
                # Direct access
                images = demo_group[image_type][:]
            elif 'obs' in demo_group and image_type in demo_group['obs']:
                # Access through obs group
                images = demo_group['obs'][image_type][:]
            else:
                # List available options
                available_images = list(demo_group.keys())
                if 'obs' in demo_group:
                    obs_images = list(demo_group['obs'].keys())
                    available_images.extend([f"obs/{img}" for img in obs_images])
                print(f"❌ Image type '{image_type}' not found. Available types: {available_images}")
                return None
            print(f"✅ Loaded {len(images)} images from demo_{demo_id}, type: {image_type}")
            print(f"   Image shape: {images.shape}")
            
            return images
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def images_to_video(images: np.ndarray, output_path: str, fps: int = 30, 
                   add_frame_numbers: bool = True, add_timestamp: bool = True) -> bool:
    """
    Convert image array to video file
    
    Args:
        images: Image array (N, H, W, 3)
        output_path: Output video file path
        fps: Frames per second
        add_frame_numbers: Whether to add frame numbers to images
        add_timestamp: Whether to add timestamp to images
        
    Returns:
        True if successful, False otherwise
    """
    if len(images) == 0:
        print("❌ No images to process")
        return False
    
    # Get video properties
    num_frames, height, width, channels = images.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Failed to create video writer for {output_path}")
        return False
    
    print(f"🎬 Creating video: {num_frames} frames, {width}x{height}, {fps} FPS")
    
    for i, img in enumerate(images):
        # Convert to BGR for OpenCV (assuming input is RGB)
        if channels == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        # Add frame number and timestamp if requested
        if add_frame_numbers or add_timestamp:
            # Convert to uint8 if needed
            if img_bgr.dtype != np.uint8:
                img_bgr = (img_bgr * 255).astype(np.uint8)
            
            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)  # White
            thickness = 2
            
            y_offset = 30
            
            if add_frame_numbers:
                frame_text = f"Frame: {i}/{num_frames-1}"
                cv2.putText(img_bgr, frame_text, (10, y_offset), font, font_scale, color, thickness)
                y_offset += 30
            
            if add_timestamp:
                timestamp = i / fps
                time_text = f"Time: {timestamp:.2f}s"
                cv2.putText(img_bgr, time_text, (10, y_offset), font, font_scale, color, thickness)
        
        # Write frame
        out.write(img_bgr)
        
        # Progress indicator
        if (i + 1) % 100 == 0 or i == num_frames - 1:
            print(f"   Processed {i + 1}/{num_frames} frames")
    
    # Release video writer
    out.release()
    
    print(f"✅ Video saved to: {output_path}")
    return True

def inspect_robomimic_file(hdf5_path: str):
    """
    Inspect RoboMimic HDF5 file structure
    
    Args:
        hdf5_path: Path to RoboMimic HDF5 file
    """
    print(f"🔍 Inspecting RoboMimic file: {hdf5_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            print(f"📁 File structure:")
            
            # List top-level groups
            for key in f.keys():
                print(f"  - {key}")
                
                if key == 'data':
                    # List demos
                    demos = [k for k in f[key].keys() if k.startswith('demo_')]
                    print(f"    📊 Available demos: {demos}")
                    
                    if demos:
                        # Inspect first demo
                        first_demo = f[key][demos[0]]
                        print(f"    🔍 Demo {demos[0]} contents:")
                        
                        for demo_key in first_demo.keys():
                            if hasattr(first_demo[demo_key], 'shape'):
                                shape = first_demo[demo_key].shape
                                dtype = first_demo[demo_key].dtype
                                print(f"      - {demo_key}: {shape} ({dtype})")
                            elif hasattr(first_demo[demo_key], 'keys'):
                                # It's a group, show its contents
                                print(f"      - {demo_key}: <group>")
                                for sub_key in first_demo[demo_key].keys():
                                    if hasattr(first_demo[demo_key][sub_key], 'shape'):
                                        shape = first_demo[demo_key][sub_key].shape
                                        dtype = first_demo[demo_key][sub_key].dtype
                                        print(f"        - {sub_key}: {shape} ({dtype})")
                                    else:
                                        print(f"        - {sub_key}: {type(first_demo[demo_key][sub_key])}")
                            else:
                                print(f"      - {demo_key}: {type(first_demo[demo_key])}")
                        
                        # Check for attributes
                        if hasattr(first_demo, 'attrs') and first_demo.attrs:
                            print(f"    📋 Demo attributes:")
                            for attr_key, attr_value in first_demo.attrs.items():
                                print(f"      - {attr_key}: {attr_value}")
                
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert RoboMimic HDF5 data to video")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to RoboMimic HDF5 file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output video file path"
    )
    
    parser.add_argument(
        "--demo_id",
        type=int,
        default=0,
        help="Demo ID to extract (default: 0)"
    )
    
    parser.add_argument(
        "--image_type",
        type=str,
        default="front_img_1_masked",
        choices=["front_img_1", "front_img_1_masked", "front_img_1_line", "front_img_1_mask"],
        help="Type of image to extract (default: front_img_1_masked)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS (default: 30)"
    )
    
    parser.add_argument(
        "--no_frame_numbers",
        action="store_true",
        help="Don't add frame numbers to video"
    )
    
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Don't add timestamp to video"
    )
    
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only inspect file structure, don't create video"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return 1
    
    # Inspect file if requested
    if args.inspect:
        inspect_robomimic_file(args.input)
        return 0
    
    # Check if output is provided for non-inspect mode
    if not args.output:
        print("❌ Output path is required when not using --inspect mode")
        return 1
    
    # Load image data
    print(f"📥 Loading data from {args.input}")
    images = load_robomimic_data(args.input, args.demo_id, args.image_type)
    
    if images is None:
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to video
    print(f"🎬 Converting to video...")
    success = images_to_video(
        images, 
        args.output, 
        fps=args.fps,
        add_frame_numbers=not args.no_frame_numbers,
        add_timestamp=not args.no_timestamp
    )
    
    if success:
        print(f"✅ Successfully created video: {args.output}")
        return 0
    else:
        print(f"❌ Failed to create video")
        return 1

if __name__ == "__main__":
    exit(main())
