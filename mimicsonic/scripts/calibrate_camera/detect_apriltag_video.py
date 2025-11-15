#!/usr/bin/env python3
"""
AprilTag检测工具 - 用于检测calibration.mp4中的tagcustom48h12标签位置
使用方法: python detect_apriltag_video.py --video_path calibration.mp4
"""

import cv2
import numpy as np
import argparse
import json
import os
import sys
from tqdm import tqdm
from datetime import datetime

# 添加项目路径以导入apriltag_detector
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from apriltag_detector import AprilTagDetector

# 添加项目路径以导入mimicsonic模块
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)


from mimicsonic.utils.mimicsonicUtils import REALSENSE_INTRINSICS



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='检测视频中的AprilTag标签位置')
    
    parser.add_argument(
        "--video_path",
        type=str,
        default="calibration.mp4",
        help="输入视频文件路径 (默认: calibration.mp4)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="calibration_detected.avi",
        help="输出视频文件路径 (默认: calibration_detected.avi)"
    )
    
    parser.add_argument(
        "--log_path",
        type=str,
        default="detection_log.json",
        help="检测日志文件路径 (默认: detection_log.json)"
    )
    
    parser.add_argument(
        "--tag_size",
        type=float,
        default=0.08,
        help="AprilTag物理尺寸，单位：米 (默认: 0.06)"
    )
    
    parser.add_argument(
        "--tag_family",
        type=str,
        default="tag36h11",
        choices=["tag36h11", "tag25h9", "tag16h5", "tagstandard41h12", "tagstandard52h13"],
        help="AprilTag标签家族 (默认: tag36h11)"
    )
    
    parser.add_argument(
        "--quad_decimate",
        type=float,
        default=1.0,
        help="四边形检测降采样因子 (默认: 1.0)"
    )
    
    parser.add_argument(
        "--quad_sigma",
        type=float,
        default=0.0,
        help="四边形检测高斯模糊sigma (默认: 0.0)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="输出视频帧率 (默认: 30)"
    )
    
    parser.add_argument(
        "--show_preview",
        action="store_true",
        help="显示实时预览窗口"
    )
    
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="保存检测失败的帧图像"
    )
    
    return parser.parse_args()


def draw_detection_info(frame, detections, frame_idx, fps):
    """在帧上绘制检测信息"""
    vis_frame = frame.copy()
    
    # 绘制帧信息
    cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Time: {frame_idx/fps:.2f}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if detections:
        cv2.putText(vis_frame, f"Tags Found: {len(detections)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制每个检测结果
        for i, detection in enumerate(detections):
            # 绘制边界框
            corners = detection.corners.astype(np.int32)
            cv2.polylines(vis_frame, [corners], True, (0, 255, 0), 2)
            
            # 绘制角点
            for corner in corners:
                cv2.circle(vis_frame, tuple(corner), 3, (255, 0, 0), -1)
            
            # 绘制中心点
            center = detection.center.astype(np.int32)
            cv2.circle(vis_frame, tuple(center), 5, (0, 0, 255), -1)
            
            # 绘制标签ID和位置信息
            tag_info = f"ID:{detection.tag_id} ({center[0]},{center[1]})"
            cv2.putText(vis_frame, tag_info, (center[0] - 50, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 绘制3D位置信息
            if hasattr(detection, 'pose_t') and detection.pose_t is not None:
                # 将numpy数组元素转换为Python float以避免格式化错误
                x = float(detection.pose_t[0])
                y = float(detection.pose_t[1])
                z = float(detection.pose_t[2])
                pos_text = f"3D: ({x:.3f}, {y:.3f}, {z:.3f})"
                cv2.putText(vis_frame, pos_text, (center[0] - 80, center[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    else:
        cv2.putText(vis_frame, "No Tags Detected", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return vis_frame


def save_detection_log(detections_log, output_path):
    """保存检测日志到JSON文件"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "total_frames": len(detections_log),
        "frames_with_detections": sum(1 for frame_data in detections_log if frame_data["detections"]),
        "total_detections": sum(len(frame_data["detections"]) for frame_data in detections_log),
        "detection_details": detections_log
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"检测日志已保存到: {output_path}")


def detect_apriltags_in_video(video_path, output_path, log_path, tag_size, tag_family, quad_decimate, quad_sigma, fps, show_preview, save_frames):
    """在视频中检测AprilTag标签"""
    
    # 检查输入文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 创建AprilTag检测器
    print(f"初始化AprilTag检测器 ({tag_family})...")
    try:
        detector = AprilTagDetector(
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            tag_family=tag_family
        )
        print(f"AprilTag检测器初始化成功 (标签家族: {tag_family})")
    except Exception as e:
        print(f"AprilTag检测器初始化失败: {e}")
        print("尝试使用默认检测器...")
        try:
            detector = AprilTagDetector(quad_decimate=quad_decimate)
            print("使用默认检测器成功")
        except Exception as e2:
            print(f"默认检测器也失败: {e2}")
            raise RuntimeError("无法初始化AprilTag检测器")
    
    # 设置相机内参
    intrinsics = {
        "fx": REALSENSE_INTRINSICS[0, 0],
        "fy": REALSENSE_INTRINSICS[1, 1],
        "cx": REALSENSE_INTRINSICS[0, 2],
        "cy": REALSENSE_INTRINSICS[1, 2],
    }
    
    print(f"相机内参: {intrinsics}")
    print(f"标签尺寸: {tag_size}m")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {video_fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames/video_fps:.2f} 秒")
    
    # 测试读取第一帧
    print("测试读取第一帧...")
    ret, test_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频的第一帧，请检查视频文件是否损坏")
    print(f"成功读取第一帧，尺寸: {test_frame.shape}")
    
    # 重置视频到开始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 设置输出视频写入器
    # 尝试不同的编码器
    fourcc_options = [
        cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'XVID'),
        cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.VideoWriter_fourcc(*'X264')
    ]
    
    out = None
    for fourcc in fourcc_options:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"成功创建视频写入器，使用编码器: {fourcc}")
            break
        else:
            out.release()
    
    if out is None or not out.isOpened():
        raise RuntimeError(f"无法创建输出视频文件: {output_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建失败帧保存目录
    if save_frames:
        fail_frames_dir = "failed_detection_frames"
        if not os.path.exists(fail_frames_dir):
            os.makedirs(fail_frames_dir)
    
    # 检测结果日志
    detections_log = []
    
    print("\n开始检测AprilTag...")
    print(f"视频文件路径: {video_path}")
    print(f"输出文件路径: {output_path}")
    
    frame_idx = 0
    detection_count = 0
    
    # 创建进度条
    pbar = tqdm(total=total_frames, desc="处理视频帧")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\n视频读取结束，共处理 {frame_idx} 帧")
                break
            
            # 初始化检测结果
            detections = []
            
            # 检测AprilTag
            try:
                detections = detector.detect(frame, intrinsics, tag_size=tag_size)
                detection_count += len(detections)
                
                # 记录检测结果
                frame_data = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / video_fps,
                    "detections": []
                }
                
                for detection in detections:
                    detection_data = {
                        "tag_id": int(detection.tag_id),
                        "center": detection.center.tolist(),
                        "corners": detection.corners.tolist(),
                        "decision_margin": float(detection.decision_margin)
                    }
                    
                    # 添加3D位置信息
                    if hasattr(detection, 'pose_t') and detection.pose_t is not None:
                        detection_data["pose_3d"] = {
                            "translation": detection.pose_t.flatten().tolist(),
                            "rotation": detection.pose_R.tolist()
                        }
                    
                    frame_data["detections"].append(detection_data)
                
                detections_log.append(frame_data)
                
                # 保存检测失败的帧
                if save_frames and len(detections) == 0:
                    fail_frame_path = os.path.join(fail_frames_dir, f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(fail_frame_path, frame)
                
            except Exception as e:
                print(f"\n检测帧 {frame_idx} 时出错: {e}")
                detections_log.append({
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / video_fps,
                    "detections": [],
                    "error": str(e)
                })
                # 确保detections为空列表
                detections = []
            
            # 绘制检测信息
            vis_frame = draw_detection_info(frame, detections, frame_idx, video_fps)
            
            # 写入输出视频
            out.write(vis_frame)
            
            # 显示预览
            if show_preview:
                cv2.imshow('AprilTag Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断检测")
                    break
            
            frame_idx += 1
            pbar.update(1)
    
    finally:
        # 清理资源
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        pbar.close()
    
    # 保存检测日志
    save_detection_log(detections_log, log_path)
    
    # 输出统计信息
    frames_with_detections = sum(1 for frame_data in detections_log if frame_data["detections"])
    
    print(f"\n检测完成!")
    print(f"处理帧数: {frame_idx}")
    print(f"检测到标签的帧数: {frames_with_detections}")
    print(f"总检测次数: {detection_count}")
    print(f"检测成功率: {frames_with_detections/frame_idx*100:.1f}%")
    print(f"输出视频: {output_path}")
    print(f"检测日志: {log_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("AprilTag视频检测工具")
    print("=" * 50)
    print(f"输入视频: {args.video_path}")
    print(f"输出视频: {args.output_path}")
    print(f"日志文件: {args.log_path}")
    print(f"标签家族: {args.tag_family}")
    print(f"标签尺寸: {args.tag_size}m")
    print(f"降采样因子: {args.quad_decimate}")
    print(f"高斯模糊sigma: {args.quad_sigma}")
    print(f"输出帧率: {args.fps} FPS")
    print("=" * 50)
    
    try:
        detect_apriltags_in_video(
            video_path=args.video_path,
            output_path=args.output_path,
            log_path=args.log_path,
            tag_size=args.tag_size,
            tag_family=args.tag_family,
            quad_decimate=args.quad_decimate,
            quad_sigma=args.quad_sigma,
            fps=args.fps,
            show_preview=args.show_preview,
            save_frames=args.save_frames
        )
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
