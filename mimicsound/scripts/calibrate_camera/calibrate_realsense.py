import os

folder_path = os.path.join(os.path.dirname(__file__))

import numpy as np
import cv2
import argparse
import json
import h5py
from tqdm import tqdm

from mimicsound.utils.mimicsoundUtils import (
    # WIDE_LENS_ROBOT_LEFT_K,
    # WIDE_LENS_ROBOT_LEFT_D,
    REALSENSE_INTRINSICS
)

from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from apriltag_detector import AprilTagDetector_my as AprilTagDetector


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h5py-path",
        type=str,
    )

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--store-matrix", action="store_true")

    return parser.parse_args()


def store_matrix(path, R, t):
    file = h5py.File(path, "r+")

    for demo_name in file.keys():
        demo = file[demo_name]
        calib_matrix_group = demo.create_group("calibration_matrix")
        calib_matrix_group.create_dataset("rotation", data=R)
        calib_matrix_group.create_dataset("translation", data=t)

    print("Appended calibration matrix: ")
    print(R.round(3))
    print(t.round(3))
    print("==============================")


def main():
    args = parse_args()

    calib = h5py.File(args.h5py_path, "r+")

    # april_detector = AprilTagDetector(quad_decimate=1.0)
    april_detector = AprilTagDetector(
    quad_decimate=1.0, 
    tag_family='tag36h11'
)

    # TODO get intrinsics
    # with open(os.path.join(args.config_folder, f"camera_{args.camera_id}_{args.camera_type}.json"), "r") as f:
    #     intrinsics = json.load(f)
    # TODO: THESE ARE JUST TEMP VALUES
    intrinsics = REALSENSE_INTRINSICS
    intrinsics = {
        "color": {
            "fx": intrinsics[0, 0],
            "fy": intrinsics[1, 1],
            "cx": intrinsics[0, 2],
            "cy": intrinsics[1, 2],
        }
    }

    print(intrinsics)

    R_base2gripper_list = []
    t_base2gripper_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    calib = calib["data"]
    count = 0
    for key in calib.keys():
        demo = calib[key]
        T, H, W, _ = demo["obs/front_img_1"].shape
        for t in tqdm(range(T)):

            img = demo["obs/front_img_1"][t]


            detect_result = april_detector.detect(
                img,
                intrinsics=intrinsics["color"],
                # tag_size=0.0958)
                # tag_size=0.17541875,
                tag_size=0.08,  # @gnq
            )

            if len(detect_result) != 1:
                print(f"wrong detection, skipping img {t}")
                if args.debug:
                    plt.imsave(f"calibration_imgs_debug/{t}_fail.png", img)
                continue

            bounding_box_corners = detect_result[0].corners
            # draw bounding box on img and save
            if args.debug:
                img = april_detector.vis_tag(img)
                plt.imsave(f"calibration_imgs_debug/{t}_detection.png", img)

            count += 1
            pose = demo["obs/ee_pose_robot_frame"][t]
            assert pose.shape == (7,)
            pos = pose[0:3]
            rot = Rot.from_quat(pose[3:])

            R_base2gripper_list.append(rot.as_matrix().T)
            t_base2gripper_list.append(
                -rot.as_matrix().T @ np.array(pos)[:, np.newaxis]
            )

            R_target2cam_list.append(detect_result[0].pose_R)
            pose_t = detect_result[0].pose_t

            # if args.debug:
            #     print("Detected: ", pose_t, T.quat2axisangle(T.mat2quat(detect_result[0].pose_R)))

            t_target2cam_list.append(pose_t)

    print(f"==========Using {count} images================")

    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
        (cv2.CALIB_HAND_EYE_PARK, "PARK"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
        (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD")
    ]
    
    results = {}
    
    for method, method_name in methods:
        try:
            R, t = cv2.calibrateHandEye(
                R_base2gripper_list,
                t_base2gripper_list,
                R_target2cam_list,
                t_target2cam_list,
                method=method,
            )
            # print("Rotation matrix: ", R.round(3))
            # print("Axis Angle: ", T.quat2axisangle(T.mat2quat(R)))
            # print("Quaternion: ", T.mat2quat(R))
            # print("Translation: ", t.T.round(3))
            fullT = np.concatenate((R, t), axis=1)
            fullT = np.concatenate((fullT, np.array([[0, 0, 0, 1]])), axis=0)
            print(f"=== {method_name} 方法结果 ===")
            print("T: ", repr(fullT))
            results[method_name] = fullT
        except Exception as e:
            print(f"=== {method_name} 方法失败 ===")
            print(f"错误: {e}")
            results[method_name] = None

    print("==============================")
    
    # 比较不同方法的结果
    print("\n=== 方法比较分析 ===")
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        method_names = list(valid_results.keys())
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                T1, T2 = valid_results[method1], valid_results[method2]
                
                # 计算平移差异
                t_diff = T1[:3, 3] - T2[:3, 3]
                translation_diff = np.linalg.norm(t_diff)
                
                # 计算旋转差异
                R_diff = T1[:3, :3] - T2[:3, :3]
                rotation_diff = np.linalg.norm(R_diff, 'fro')
                
                print(f"{method1} vs {method2}:")
                print(f"  平移差异: {translation_diff:.4f} 米")
                print(f"  旋转差异: {rotation_diff:.4f}")
                print()

    if args.store_matrix:
        # 使用TSAI方法的结果作为默认保存
        if 'TSAI' in valid_results:
            R = valid_results['TSAI'][:3, :3]
            t = valid_results['TSAI'][:3, 3:4]
            store_matrix(args.h5py_path, R, t.T)
        else:
            print("警告: 没有有效的标定结果可以保存")


if __name__ == "__main__":
    main()
