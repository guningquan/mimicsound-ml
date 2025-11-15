#!/usr/bin/env python3
"""
从MimicSonic HDF5数据集中提取图像序列并创建视频
支持front_img_1、front_img_1_masked、front_img_1_line等多种图像类型
"""

import h5py
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

def create_video_from_hdf5(hdf5_path, output_path, fps=30, demo_idx=0, max_frames=None):
    """
    从HDF5文件中提取图像序列并创建视频
    
    Args:
        hdf5_path (str): HDF5文件路径
        output_path (str): 输出视频文件路径
        fps (int): 视频帧率
        demo_idx (int): 要处理的demo索引
        max_frames (int): 最大帧数,None表示处理所有帧
    """
    
    print(f"正在读取HDF5文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 检查数据结构
        if 'data' not in f:
            raise ValueError("HDF5文件中没有找到'data'组")
        
        data_group = f['data']
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        
        if not demo_keys:
            raise ValueError("HDF5文件中没有找到demo数据")
        
        # 选择要处理的demo
        if demo_idx >= len(demo_keys):
            print(f"警告: demo_idx {demo_idx} 超出范围，使用 demo_0")
            demo_idx = 0
        
        demo_key = demo_keys[demo_idx]
        demo_group = data_group[demo_key]
        
        print(f"处理demo: {demo_key}")
        
        # 检查obs组
        if 'obs' not in demo_group:
            raise ValueError(f"Demo {demo_key} 中没有找到'obs'组")
        
        obs_group = demo_group['obs']
        
        # 检查front_img_1
        if 'front_img_1' not in obs_group:
            raise ValueError(f"Demo {demo_key} 中没有找到'front_img_1'图像数据")
        
        front_img_data = obs_group['front_img_1']
        print(f"图像数据形状: {front_img_data.shape}")
        print(f"图像数据类型: {front_img_data.dtype}")
        
        # 获取图像尺寸
        if len(front_img_data.shape) == 4:  # (T, H, W, C)
            num_frames, height, width, channels = front_img_data.shape
        else:
            raise ValueError(f"不支持的图像数据形状: {front_img_data.shape}")
        
        # 限制帧数
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
        
        print(f"将处理 {num_frames} 帧图像")
        print(f"图像尺寸: {width}x{height}")
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {output_path}")
        
        print(f"开始创建视频: {output_path}")
        
        # 逐帧处理图像
        for frame_idx in tqdm(range(num_frames), desc="处理帧"):
            # 获取当前帧
            frame = front_img_data[frame_idx]
            
            # 确保数据类型和范围正确
            if frame.dtype != np.uint8:
                # 如果是float类型，假设范围是[0,1]，转换为[0,255]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # 确保图像是BGR格式（OpenCV要求）
            if channels == 3:
                # 假设输入是RGB，转换为BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 写入视频
            video_writer.write(frame)
        
        # 释放资源
        video_writer.release()
        
        print(f"视频创建完成: {output_path}")
        print(f"总帧数: {num_frames}")
        print(f"帧率: {fps} FPS")
        print(f"视频时长: {num_frames/fps:.2f} 秒")

def create_multiple_videos_from_hdf5(hdf5_path, output_dir, fps=30, demo_idx=0, max_frames=None, image_types=None):
    """
    从HDF5文件中提取多种图像类型并创建对应的视频
    
    Args:
        hdf5_path (str): HDF5文件路径
        output_dir (str): 输出目录路径
        fps (int): 视频帧率
        demo_idx (int): 要处理的demo索引
        max_frames (int): 最大帧数,None表示处理所有帧
        image_types (list): 要处理的图像类型列表，None表示自动检测
    """
    
    print(f"正在读取HDF5文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 检查数据结构
        if 'data' not in f:
            raise ValueError("HDF5文件中没有找到'data'组")
        
        data_group = f['data']
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        
        if not demo_keys:
            raise ValueError("HDF5文件中没有找到demo数据")
        
        # 选择要处理的demo
        if demo_idx >= len(demo_keys):
            print(f"警告: demo_idx {demo_idx} 超出范围，使用 demo_0")
            demo_idx = 0
        
        demo_key = demo_keys[demo_idx]
        demo_group = data_group[demo_key]
        
        print(f"处理demo: {demo_key}")
        
        # 检查obs组
        if 'obs' not in demo_group:
            raise ValueError(f"Demo {demo_key} 中没有找到'obs'组")
        
        obs_group = demo_group['obs']
        
        # 自动检测可用的图像类型
        if image_types is None:
            image_types = []
            # 检查front相关的图像类型
            front_types = ['front_img_1', 'front_img_1_masked', 'front_img_1_line']
            for img_type in front_types:
                if img_type in obs_group:
                    image_types.append(img_type)
        
        if not image_types:
            raise ValueError(f"Demo {demo_key} 中没有找到任何可用的图像数据")
        
        print(f"将处理以下图像类型: {image_types}")
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 为每种图像类型创建视频
        for img_type in image_types:
            print(f"\n=== 处理图像类型: {img_type} ===")
            
            img_data = obs_group[img_type]
            print(f"图像数据形状: {img_data.shape}")
            print(f"图像数据类型: {img_data.dtype}")
            
            # 获取图像尺寸
            if len(img_data.shape) == 4:  # (T, H, W, C)
                num_frames, height, width, channels = img_data.shape
            else:
                print(f"警告: 不支持的图像数据形状: {img_data.shape}，跳过")
                continue
            
            # 限制帧数
            if max_frames is not None:
                num_frames = min(num_frames, max_frames)
            
            print(f"将处理 {num_frames} 帧图像")
            print(f"图像尺寸: {width}x{height}")
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
            output_filename = f"{base_name}_{demo_key}_{img_type}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                print(f"警告: 无法创建视频文件: {output_path}，跳过")
                continue
            
            print(f"开始创建视频: {output_path}")
            
            # 逐帧处理图像
            for frame_idx in tqdm(range(num_frames), desc=f"处理{img_type}帧"):
                # 获取当前帧
                frame = img_data[frame_idx]
                
                # 确保数据类型和范围正确
                if frame.dtype != np.uint8:
                    # 如果是float类型，假设范围是[0,1]，转换为[0,255]
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # 确保图像是BGR格式（OpenCV要求）
                if channels == 3:
                    # 假设输入是RGB，转换为BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 写入视频
                video_writer.write(frame)
            
            # 释放资源
            video_writer.release()
            
            print(f"视频创建完成: {output_path}")
            print(f"总帧数: {num_frames}")
            print(f"帧率: {fps} FPS")
            print(f"视频时长: {num_frames/fps:.2f} 秒")
        
        print(f"\n所有视频创建完成！输出目录: {output_dir}")

def create_concatenated_video_from_hdf5(hdf5_path, output_path, fps=30, demo_idx=0, max_frames=None):
    """
    从HDF5文件中提取front_img_1、front_img_1_masked、front_img_1_line图像
    并创建水平拼接的1*3视频
    
    Args:
        hdf5_path (str): HDF5文件路径
        output_path (str): 输出视频文件路径
        fps (int): 视频帧率
        demo_idx (int): 要处理的demo索引
        max_frames (int): 最大帧数,None表示处理所有帧
    """
    
    print(f"正在读取HDF5文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 检查数据结构
        if 'data' not in f:
            raise ValueError("HDF5文件中没有找到'data'组")
        
        data_group = f['data']
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        
        if not demo_keys:
            raise ValueError("HDF5文件中没有找到demo数据")
        
        # 选择要处理的demo
        if demo_idx >= len(demo_keys):
            print(f"警告: demo_idx {demo_idx} 超出范围，使用 demo_0")
            demo_idx = 0
        
        demo_key = demo_keys[demo_idx]
        demo_group = data_group[demo_key]
        
        print(f"处理demo: {demo_key}")
        
        # 检查obs组
        if 'obs' not in demo_group:
            raise ValueError(f"Demo {demo_key} 中没有找到'obs'组")
        
        obs_group = demo_group['obs']
        
        # 自动检测可用的图像类型
        print(f"Demo {demo_key} 中可用的图像类型:")
        print(f"obs_group.keys(): {list(obs_group.keys())}")
        
        available_types = []
        for key in obs_group.keys():
            if key.startswith('front_img_1'):
                available_types.append(key)
                try:
                    data_shape = obs_group[key].shape
                    data_dtype = obs_group[key].dtype
                    print(f"  - {key}: shape={data_shape}, dtype={data_dtype}")
                except Exception as e:
                    print(f"  - {key}: 访问错误 - {e}")
        
        print(f"找到的front_img_1相关图像类型: {available_types}")
        
        # 优先选择的图像类型（按优先级排序）
        preferred_types = ['front_img_1', 'front_img_1_masked', 'front_img_1_line']
        selected_types = []
        
        # 按优先级选择可用的图像类型
        for pref_type in preferred_types:
            if pref_type in available_types:
                selected_types.append(pref_type)
        
        if not selected_types:
            raise ValueError(f"Demo {demo_key} 中没有找到任何front_img_1相关的图像数据")
        
        print(f"将使用以下图像类型进行拼接: {selected_types}")
        
        # 获取选中的图像数据并验证
        image_data = {}
        for img_type in selected_types:
            try:
                print(f"正在访问图像数据: {img_type}")
                data = obs_group[img_type]
                print(f"成功访问 {img_type}: shape={data.shape}, dtype={data.dtype}")
                
                # 验证数据形状
                if len(data.shape) != 4:
                    raise ValueError(f"图像数据 {img_type} 的形状不正确: {data.shape}，期望4维 (T, H, W, C)")
                
                # 预览第一帧数据
                first_frame = data[0]
                print(f"{img_type} 第一帧: shape={first_frame.shape}, dtype={first_frame.dtype}, range=[{first_frame.min()}, {first_frame.max()}]")
                
                image_data[img_type] = data
                print(f"{img_type} 数据验证通过")
                
            except Exception as e:
                print(f"访问图像数据 {img_type} 时出错: {e}")
                raise
        
        # 获取图像尺寸（假设所有图像尺寸相同）
        first_img = image_data['front_img_1']
        if len(first_img.shape) == 4:  # (T, H, W, C)
            num_frames, height, width, channels = first_img.shape
        else:
            raise ValueError(f"不支持的图像数据形状: {first_img.shape}")
        
        # 限制帧数
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
        
        print(f"将处理 {num_frames} 帧图像")
        print(f"单个图像尺寸: {width}x{height}")
        print(f"拼接后尺寸: {width*len(selected_types)}x{height}")
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width*len(selected_types), height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {output_path}")
        
        print(f"开始创建拼接视频: {output_path}")
        
        # 逐帧处理图像
        print(f"开始处理 {num_frames} 帧图像...")
        for frame_idx in tqdm(range(num_frames), desc="处理拼接帧"):
            # 获取所有选中图像类型的当前帧
            frames = []
            for img_type in selected_types:
                try:
                    # 获取当前帧
                    frame = image_data[img_type][frame_idx]
                    
                    # 调试信息（仅第一帧）
                    if frame_idx == 0:
                        print(f"  {img_type} 第{frame_idx}帧: shape={frame.shape}, dtype={frame.dtype}, range=[{frame.min()}, {frame.max()}]")
                    
                    # 确保数据类型和范围正确
                    if frame.dtype != np.uint8:
                        # 如果是float类型，假设范围是[0,1]，转换为[0,255]
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    
                    # 确保图像是BGR格式（OpenCV要求）
                    if channels == 3:
                        # 假设输入是RGB，转换为BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    frames.append(frame)
                    
                except Exception as e:
                    print(f"处理 {img_type} 第{frame_idx}帧时出错: {e}")
                    raise
            
            # 水平拼接所有图像
            try:
                concatenated_frame = np.concatenate(frames, axis=1)
                
                # 调试信息（仅第一帧）
                if frame_idx == 0:
                    print(f"  拼接后帧: shape={concatenated_frame.shape}, dtype={concatenated_frame.dtype}")
                
                # 写入视频
                video_writer.write(concatenated_frame)
                
            except Exception as e:
                print(f"拼接第{frame_idx}帧时出错: {e}")
                raise
        
        # 释放资源
        video_writer.release()
        
        print(f"拼接视频创建完成: {output_path}")
        print(f"总帧数: {num_frames}")
        print(f"帧率: {fps} FPS")
        print(f"视频时长: {num_frames/fps:.2f} 秒")
        print(f"拼接布局: 1*{len(selected_types)} ({' | '.join(selected_types)})")

def main():
    parser = argparse.ArgumentParser(description='从MimicSonic HDF5数据集创建视频')
    parser.add_argument('--hdf5_path', type=str, 
                       default='/home/robot/Dataset_and_Checkpoint/mimicsonic/datasets/timer_Oct10.hdf5',
                       help='HDF5文件路径')
    parser.add_argument('--output_path', type=str, 
                       default='timer_robot_demo.mp4',
                       help='输出视频文件路径（单视频模式）')
    parser.add_argument('--output_dir', type=str, 
                       default='./videos',
                       help='输出目录路径（多视频模式）')
    parser.add_argument('--fps', type=int, default=30,
                       help='视频帧率')
    parser.add_argument('--demo_idx', type=int, default=0,
                       help='要处理的demo索引')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='最大帧数, None表示处理所有帧')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple', 'concatenated'], default='concatenated',
                       help='创建模式: single=单个视频, multiple=多个视频, concatenated=拼接视频（1*3布局）')
    parser.add_argument('--image_types', type=str, nargs='+', default=None,
                       help='指定要处理的图像类型（多视频模式）')
    parser.add_argument('--concatenated_output', type=str, 
                       default='concatenated_video.mp4',
                       help='拼接视频输出文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.hdf5_path):
        print(f"错误: HDF5文件不存在: {args.hdf5_path}")
        return
    
    try:
        if args.mode == 'single':
            # 单视频模式
            # 创建输出目录
            output_dir = os.path.dirname(args.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            create_video_from_hdf5(
                hdf5_path=args.hdf5_path,
                output_path=args.output_path,
                fps=args.fps,
                demo_idx=args.demo_idx,
                max_frames=args.max_frames
            )
        elif args.mode == 'multiple':
            # 多视频模式
            create_multiple_videos_from_hdf5(
                hdf5_path=args.hdf5_path,
                output_dir=args.output_dir,
                fps=args.fps,
                demo_idx=args.demo_idx,
                max_frames=args.max_frames,
                image_types=args.image_types
            )
        else:  # concatenated mode
            # 拼接视频模式
            # 创建输出目录
            output_dir = os.path.dirname(args.concatenated_output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            create_concatenated_video_from_hdf5(
                hdf5_path=args.hdf5_path,
                output_path=args.concatenated_output,
                fps=args.fps,
                demo_idx=args.demo_idx,
                max_frames=args.max_frames
            )
    except Exception as e:
        print(f"错误: {e}")
        return

if __name__ == '__main__':
    main()
