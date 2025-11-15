import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm
from mimicsonic.utils.mimicsonicUtils import (
    nds,
    ee_pose_to_cam_frame,
    EXTRINSICS,
    AlohaFK,
    REALSENSE_INTRINSICS
)
from mimicsonic.scripts.aloha_process.aloha_scripts.constants import DT
import pytorch_kinematics as pk
import torch
import scipy.signal
from scipy.io import wavfile

# from modern_robotics import FKinSpace
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
import json

# from external.robomimic.robomimic.utils.dataset import interpolate_arr
# Temporarily commented out as this function may not be needed
from mimicsonic.scripts.masking.utils import *

# POINT_GAP = 2  # @gnq important
POINT_GAP = 1

"""
aloha_hdf5 has the following format
dict with keys:  <KeysViewHDF5 ['action', 'observations']>
action: (500, 14)
observations: dict with keys:  <KeysViewHDF5 ['effort', 'images', 'qpos', 'qvel']>
        effort: (500, 14)
        images: dict with keys:  <KeysViewHDF5 ['cam_high', 'cam_right_wrist']>
                cam_high: (500, 480, 640, 3)
                cam_right_wrist: (500, 480, 640, 3)
        qpos: (500, 14)
        qvel: (500, 14)
"""


def process_audio_data(demo_hdf5, num_frames, audio_length=2.0, target_sampling_rate=16000, DT=0.02):
    """
    Process audio data from the demo HDF5 file by concatenating previous timesteps.
    
    Args:
        demo_hdf5: HDF5 file object containing the demo data
        num_frames: Number of video frames (for alignment)
        audio_length: Desired audio length in seconds
        target_sampling_rate: Target sampling rate in Hz
        DT: Time step duration in seconds
    
    Returns:
        processed_audio: Processed audio data aligned with video frames
    """
    if "audio" not in demo_hdf5:
        print("Warning: No audio data found in demo file")
        return None
    
    # Get original audio data and metadata
    audio_data = demo_hdf5["audio"][:]  # Shape: (num_timesteps, max_samples)
    original_sampling_rate = demo_hdf5.attrs.get("audio_sampling_rate", 48000)
    audio_channels = demo_hdf5.attrs.get("audio_channels", 1)
    
    print(f"Processing audio: {audio_data.shape[0]} timesteps, original rate: {original_sampling_rate}Hz")
    
    # Calculate audio length in timesteps
    audio_length_timesteps = int(audio_length / DT)
    max_timesteps, max_substeps_plus1 = audio_data.shape
    max_substeps = max_substeps_plus1 - 1
    
    # Calculate target audio length in samples
    max_audio_length = int(target_sampling_rate * audio_length)
    
    # Process each timestep
    processed_audio_frames = []
    
    for start_ts in range(num_frames):
        # Calculate start index for audio window
        start_idx = max(0, start_ts - (audio_length_timesteps - 1))
        
        # Select audio data for the time window
        selected_audio = []
        for ts in range(start_idx, start_ts + 1):
            if ts < max_timesteps:
                # Get actual length (stored in last element)
                valid_length = audio_data[ts, -1]
                if valid_length > 0:
                    # Extract valid audio data
                    selected_audio.append(audio_data[ts, :valid_length])
                else:
                    # No audio data for this timestep, add zeros
                    selected_audio.append(np.zeros((max_substeps,), dtype=np.int16))
            else:
                # Beyond timestep range, add zeros
                selected_audio.append(np.zeros((max_substeps,), dtype=np.int16))
        
        # Handle missing timesteps by padding with zeros at the beginning
        missing_steps = audio_length_timesteps - len(selected_audio)
        if missing_steps > 0:
            zero_padding = [np.zeros((max_substeps,), dtype=np.int16)] * missing_steps
            selected_audio = zero_padding + selected_audio
        
        # Concatenate audio data
        concatenated_audio = np.concatenate(selected_audio) if selected_audio else np.zeros(0, dtype=np.int16)
        
        # Resample if needed
        if original_sampling_rate != target_sampling_rate:
            resample_ratio = target_sampling_rate / original_sampling_rate
            new_length = int(len(concatenated_audio) * resample_ratio)
            concatenated_audio = scipy.signal.resample(concatenated_audio, new_length)
        
        # Adjust length to target
        if concatenated_audio.shape[0] > max_audio_length:
            # Truncate from the end (keep most recent audio)
            concatenated_audio = concatenated_audio[-max_audio_length:]
        elif concatenated_audio.shape[0] < max_audio_length:
            # Pad with zeros at the beginning
            padding = np.zeros((max_audio_length - concatenated_audio.shape[0],), dtype=np.float32)
            concatenated_audio = np.concatenate([padding, concatenated_audio.astype(np.float32)])
        else:
            concatenated_audio = concatenated_audio.astype(np.float32)
        
        # Normalize audio to [-1, 1] range
        concatenated_audio = concatenated_audio / 32768.0
        
        processed_audio_frames.append(concatenated_audio)
    
    # Convert to numpy array
    processed_audio = np.array(processed_audio_frames, dtype=np.float32)
    
    print(f"Processed audio shape: {processed_audio.shape}")
    return processed_audio


def get_future_points(arr, POINT_GAP=15, FUTURE_POINTS_COUNT=10):
    """
    arr: (T, ACTION_DIM)
    POINT_GAP: how many timesteps to skip
    FUTURE_POINTS_COUNT: how many future points to collect
    given an array arr, prepack the future points into each timestep.  return an array of size (T, FUTURE_POINTS_COUNT, ACTION_DIM).  If there are not enough future points, pad with the last point.
    do it purely vectorized
    """
    T, ACTION_DIM = arr.shape
    result = np.zeros((T, FUTURE_POINTS_COUNT, ACTION_DIM))
    
    for t in range(T):
        future_indices = np.arange(t, t + POINT_GAP * (FUTURE_POINTS_COUNT), POINT_GAP)
        future_indices = np.clip(future_indices, 0, T - 1)
        result[t] = arr[future_indices]
    return result


def sample_interval_points(arr, POINT_GAP=15, FUTURE_POINTS_COUNT=10):
    """
    arr: (T, ACTION_DIM)
    POINT_GAP: how many timesteps to skip between points
    FUTURE_POINTS_COUNT: how many future points to collect
    Returns an array of points sampled at intervals of POINT_GAP * FUTURE_POINTS_COUNT.
    """
    num_samples, T, ACTION_DIM = arr.shape
    interval = T / 10
    indices = np.arange(0, T, interval).astype(int)
    sampled_points = arr[:, indices, :]
    return sampled_points


def is_valid_path(path):
    return not os.path.isdir(path) and "episode" in path and ".hdf5" in path


def apply_masking(hdf5_file, arm, extrinsics):
    """
    hdf5_file: path to the hdf5 file to iterate over
    arm: arm to mask - left, right, or both
    extrinsics: which extrinsics to use
    Apply SAM-based masking and overlayed line to images in the hdf5 file.
    """
    print(".........Starting Masking........")
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()

    with h5py.File(hdf5_file, 'r+') as aloha_hdf5, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        keys_list = list(aloha_hdf5['data'].keys())
        keys_list = [k.split('_')[1] for k in keys_list]
        for j in tqdm(keys_list):
            print(f"Processing episode {j}")
            
            px_dict = sam.project_joint_positions_to_image(torch.from_numpy(aloha_hdf5[f'data/demo_{j}/obs/joint_positions'][:, :]), extrinsics, REALSENSE_INTRINSICS, arm=arm)

            mask_images, line_images = sam.get_robot_mask_line_batched(
                aloha_hdf5[f'data/demo_{j}/obs/front_img_1'], px_dict, arm=arm)


            # # 可视化第一个样本（或任意第 idx 个）
            # idx = 400  # 你可以改成 5、10 等看看不同帧
            # original_img = aloha_hdf5[f'data/demo_{j}/obs/front_img_1'][idx]
            # line_img = line_images[idx]
            # print("px_val_gripper_right: ", px_dict["px_val_gripper_right"][idx])
            # print("px_val_gripper_left: ", px_dict["px_val_gripper_left"][idx])
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
            

            if "front_img_1_line" in aloha_hdf5[f'data/demo_{j}/obs']:
                del aloha_hdf5[f'data/demo_{j}/obs/front_img_1_line']
            
            if "front_img_1_masked" in aloha_hdf5[f'data/demo_{j}/obs']:
                del aloha_hdf5[f'data/demo_{j}/obs/front_img_1_masked']

            aloha_hdf5[f'data/demo_{j}/obs'].create_dataset('front_img_1_line', data=line_images, chunks=(1, 480, 640, 3))
            aloha_hdf5[f'data/demo_{j}/obs'].create_dataset('front_img_1_masked', data=mask_images, chunks=(1, 480, 640, 3))


def decompress_image_data(compressed_data):
    """
    Decompress act-plus-plus compressed image data
    compressed_data: (T, compressed_size) compressed data
    Returns: (T, 480, 640, 3) decompressed image data
    """
    import cv2
    
    T, compressed_size = compressed_data.shape
    decompressed_images = []
    
    print(f"Decompressing {T} frames, compressed size: {compressed_size}")
    
    for i in range(T):
        # Decompress single frame image
        compressed_frame = compressed_data[i]
        
        # Use cv2 to decompress
        try:
            # Decompress image
            decompressed = cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR)
            if decompressed is not None:
                # Ensure correct image dimensions
                if decompressed.shape[:2] != (480, 640):
                    decompressed = cv2.resize(decompressed, (640, 480))
                # Convert BGR to RGB - this is the key step
                # cv2.imdecode returns BGR format, need to convert to RGB
                # decompressed_rgb = cv2.cvtColor(decompressed, cv2.COLOR_BGR2RGB)
                decompressed_rgb = decompressed
                decompressed_images.append(decompressed_rgb)

                # Display the decompressed image
                # plt.figure(figsize=(12, 8))
                # plt.subplot(2, 2, 1)
                # plt.imshow(decompressed_rgb)
                # plt.title('Decompressed RGB Image')
                # plt.axis('off')
                # plt.show()
                # plt.close()
            else:
                # If decompression fails, create black image
                print(f"Warning: Frame {i} decompression failed, using black image")
                decompressed_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
        except Exception as e:
            print(f"Warning: Frame {i} decompression error: {e}, using black image")
            decompressed_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    return np.array(decompressed_images)

def add_image_obs(demo_hdf5, demo_obs_group, cam_name):
    """
    demo_hdf5: the demo hdf5 file
    demo_obs_group: the demo obs object
    cam_name: the name of the camera to add
    Add an image to the demo hdf5 file.
    """
    if cam_name == "cam_high":
        image_data = demo_hdf5["observations"]["images"]["cam_high"]
        
        # Check data format
        if len(image_data.shape) == 2:  # Compressed format (T, compressed_size)
            print("Detected compressed image data, decompressing...")
            image_data = decompress_image_data(image_data)
        
        demo_obs_group.create_dataset(
            "front_img_1",
            data=image_data,
            dtype="uint8",
            chunks=(1, 480, 640, 3),
        )
    elif cam_name == "cam_left_wrist":
        image_data = demo_hdf5["observations"]["images"]["cam_left_wrist"]
        
        # Check data format
        if len(image_data.shape) == 2:  # Compressed format
            print("Detected compressed image data, decompressing...")
            image_data = decompress_image_data(image_data)
        
        demo_obs_group.create_dataset(
            "left_wrist_img",
            data=image_data,
            dtype="uint8",
            chunks=(1, 480, 640, 3),
        )
    elif cam_name == "cam_right_wrist":
        image_data = demo_hdf5["observations"]["images"]["cam_right_wrist"]
        
        # Check data format
        if len(image_data.shape) == 2:  # Compressed format
            print("Detected compressed image data, decompressing...")
            image_data = decompress_image_data(image_data)
        
        demo_obs_group.create_dataset(
            "right_wrist_img",
            data=image_data,
            dtype="uint8",
            chunks=(1, 480, 640, 3),
        )    

def add_joint_actions(demo_hdf5, demo_i_group, joint_start, joint_end, prestack=False, POINT_GAP=2, FUTURE_POINTS_COUNT=100):
    """
    demo_hdf5: the demo hdf5 file
    demo_i_group: the demo group to write the data to
    joint_start: the start index of the joint actions
    joint_end: the end index of the joint actions
    prestack: whether to prestack the future points
    POINT_GAP: how many timesteps to skip
    FUTURE_POINTS_COUNT: how many future points to collect

    Add joint actions to the demo hdf5 file.
    """
    joint_actions = demo_hdf5["action"][:,  joint_start:joint_end]
    if prestack:
        joint_actions = get_future_points(joint_actions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
        joint_actions_sampled = sample_interval_points(joint_actions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
    else:
        # If not prestacking, use original data
        joint_actions_sampled = joint_actions
    
    demo_i_group.create_dataset(
        "actions_joints", data=joint_actions_sampled
    )
    demo_i_group.create_dataset(
        "actions_joints_act", data=joint_actions
    )
    

def add_xyz_actions(demo_hdf5, demo_i_group, arm, left_extrinsics=None, right_extrinsics=None, prestack=False, POINT_GAP=2, FUTURE_POINTS_COUNT=100):
    """
    demo_hdf5: the demo hdf5 file
    demo_i_group: the demo group to write the data to
    arm: the arm to process
    left_extrinsics: the left camera extrinsics
    right_extrinsics: the right camera extrinsics
    prestack: whether to prestack the future points
    POINT_GAP: how many timesteps to skip
    FUTURE_POINTS_COUNT: how many future points to collect

    Add xyz actions to the demo hdf5 file.
    """
    aloha_fk = AlohaFK()

    if arm == "both":
        joint_start = 0
        joint_end = 14

        #Needed for forward kinematics
        joint_left_start = 0
        joint_left_end = 7
        joint_right_start = 7
        joint_right_end = 14
        
        fk_left_positions = aloha_fk.fk(demo_hdf5["action"][:, joint_left_start:joint_left_end - 1])
        fk_right_positions = aloha_fk.fk(demo_hdf5["action"][:, joint_right_start:joint_right_end - 1])
    else:
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        fk_positions = aloha_fk.fk(demo_hdf5["action"][:, joint_start:joint_end - 1])
    
    if arm == "both":
        fk_left_positions = ee_pose_to_cam_frame(
            fk_left_positions, left_extrinsics
        )[:, :3]
        fk_right_positions = ee_pose_to_cam_frame(
            fk_right_positions, right_extrinsics
        )[:, :3]
        fk_positions = np.concatenate([fk_left_positions, fk_right_positions], axis=1)
        # print(f"fk_positions: {fk_positions}")
        # exit()
    else:
        extrinsics = left_extrinsics if arm == "left" else right_extrinsics         
        fk_positions = ee_pose_to_cam_frame(
            fk_positions, extrinsics
        )[:, :3]

    if prestack:
        print("prestacking", fk_positions.shape)
        fk_positions = get_future_points(fk_positions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
        print("AFTER prestacking", fk_positions.shape)
        fk_positions_sampled = sample_interval_points(fk_positions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
    else:
        # If not prestacking, use original data
        fk_positions_sampled = fk_positions

    demo_i_group.create_dataset("actions_xyz_act", data=fk_positions)
    demo_i_group.create_dataset("actions_xyz", data=fk_positions_sampled)

def add_ee_pose_obs(demo_hdf5, demo_i_obs_group, arm, left_extrinsics=None, right_extrinsics=None): 
    """
    demo_hdf5: the demo hdf5 file
    demo_i_obs_group: the demo obs group to write the data to
    arm: the arm to process
    left_extrinsics: the left camera extrinsics
    right_extrinsics: the right camera extrinsics

    Add ee pose obs to the demo hdf5 file.
    """
    aloha_fk = AlohaFK()

    if arm == "both":
        joint_start = 0
        joint_end = 14
        #Needed for forward kinematics
        joint_left_start = 0
        joint_left_end = 7
        joint_right_start = 7
        joint_right_end = 14
        fk_left_positions = aloha_fk.fk(demo_hdf5["observations"]["qpos"][:, joint_left_start:joint_left_end - 1])
        fk_right_positions = aloha_fk.fk(demo_hdf5["observations"]["qpos"][:, joint_right_start:joint_right_end - 1])
    else:
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14    
        fk_positions = aloha_fk.fk(demo_hdf5["observations"]["qpos"][:, joint_start:joint_end - 1])
    
    if arm == "both":
        fk_left_positions = ee_pose_to_cam_frame(
            fk_left_positions, left_extrinsics
        )[:, :3]
        fk_right_positions = ee_pose_to_cam_frame(
            fk_right_positions, right_extrinsics
        )[:, :3]
        fk_positions = np.concatenate([fk_left_positions, fk_right_positions], axis=1)
    else:
        extrinsics = left_extrinsics if arm == "left" else right_extrinsics   
        fk_positions = ee_pose_to_cam_frame(
            fk_positions, extrinsics
        )[:, :3]

    demo_i_obs_group.create_dataset("ee_pose", data=fk_positions)

def process_demo(demo_path, data_group, arm, extrinsics, prestack=False, enable_audio=False, audio_length=2.0, target_sampling_rate=16000):
    """
    demo_path: path to the demo hdf5 file
    data_group: the group in the output hdf5 file to write the data to
    arm: arm to process - left, right, or both
    extrinsics: camera extrinsics. It is a tuple of (left_extrinsics, right_extrinsics) if arm is both
    prestack: whether to prestack the future points
    Process a single demo hdf5 file and write the data to the output hdf5 file.
    """

    left_extrinsics = None
    right_extrinsics = None
    if arm == "both":
        if not isinstance(extrinsics, dict):
            print("Error: Both arms selected. Expected extrinsics for both arms.")
        left_extrinsics = extrinsics["left"]
        right_extrinsics = extrinsics["right"]
    elif args.arm == "left":
        extrinsics = extrinsics["left"]
        left_extrinsics = extrinsics
    elif args.arm == "right":
        extrinsics = extrinsics["right"]
        right_extrinsics = extrinsics
    with h5py.File(demo_path, "r") as demo_hdf5:
        demo_number = demo_path.split("_")[-1].split(".")[0]
        demo_i_group = data_group.create_group(f"demo_{demo_number}")
        demo_i_group.attrs["num_samples"] = demo_hdf5["action"].shape[0]
        demo_i_obs_group = demo_i_group.create_group("obs")

        # Extract the data from the aloha hdf5 file
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        elif arm == "both":
            joint_start = 0
            joint_end = 14

            #Needed for forward kinematics
            joint_left_start = 0
            joint_left_end = 7
            joint_right_start = 7
            joint_right_end = 14

        # obs
        ## adding images
        add_image_obs(demo_hdf5, demo_i_obs_group, "cam_high")
        if arm in ["left", "both"]:
            add_image_obs(demo_hdf5, demo_i_obs_group, "cam_left_wrist")
        if arm in ["right", "both"]:
            add_image_obs(demo_hdf5, demo_i_obs_group, "cam_right_wrist")
        
        ## add joint obs
        demo_i_obs_group.create_dataset(
            "joint_positions", data=demo_hdf5["observations"]["qpos"][:, joint_start:joint_end]
        )

        # add ee_pose
        add_ee_pose_obs(demo_hdf5, demo_i_obs_group, arm, left_extrinsics=left_extrinsics, right_extrinsics=right_extrinsics)

        # POINT_GAP = 2  # @gnq important
        POINT_GAP = 2
        FUTURE_POINTS_COUNT = 100

        # add joint actions
        add_joint_actions(demo_hdf5, demo_i_group, joint_start, joint_end, prestack=prestack, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)

        # actions_xyz
        add_xyz_actions(demo_hdf5, demo_i_group, arm, left_extrinsics, right_extrinsics, prestack=prestack, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
        
        # Process audio data if enabled
        if enable_audio:
            num_frames = demo_hdf5["action"].shape[0]
            processed_audio = process_audio_data(
                demo_hdf5, 
                num_frames, 
                audio_length=audio_length,
                target_sampling_rate=target_sampling_rate,
                DT=DT
            )
            
            if processed_audio is not None:
                # Add audio data to observations
                demo_i_obs_group.create_dataset("audio", data=processed_audio)
                # Store audio metadata as attributes
                demo_i_obs_group.attrs["audio_sampling_rate"] = target_sampling_rate
                demo_i_obs_group.attrs["audio_length"] = audio_length
                print(f"Added audio data to demo: shape {processed_audio.shape}")
   
def  main(args):
    # before converting everything, check it all at least opens
    for file in tqdm(os.listdir(args.dataset)):
        #  if os.path.isfile(os.path.join(args.dataset, file)):
        #     print(file.split("_")[1].split(".")[0])
        #     if int(file.split("_")[1].split(".")[0]) <= 5:
        print("Trying to open " + file)
        to_open = os.path.join(args.dataset, file)
        print(to_open)
        if is_valid_path(to_open):
            with h5py.File(to_open, "r") as f:
                pass

    with h5py.File(args.out, "w", rdcc_nbytes=1024**2 * 2) as dataset:
        data_group = dataset.create_group("data")
        data_group.attrs["env_args"] = json.dumps({})  # if no normalize obs

        for i, aloha_demo in enumerate(tqdm(os.listdir(args.dataset))):
            if not is_valid_path(os.path.join(args.dataset, aloha_demo)):
                continue

            aloha_demo_path = os.path.join(args.dataset, aloha_demo)

            process_demo(
                aloha_demo_path, 
                data_group, 
                args.arm, 
                EXTRINSICS[args.extrinsics], 
                args.prestack,
                enable_audio=args.enable_audio,
                audio_length=args.audio_length,
                target_sampling_rate=args.target_sampling_rate
            )

    split_train_val_from_hdf5(hdf5_path=args.out, val_ratio=args.val_ratio, filter_key=None)

    ## Masking
    if args.mask:
        print("Starting Masking")
        apply_masking(args.out, args.arm, EXTRINSICS[args.extrinsics])
    print("Successful Conversion!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to rawAloha folder",
    )
    parser.add_argument("--arm", type=str, help="which arm to convert data for")
    parser.add_argument("--extrinsics", type=str, help="which arm to convert data for")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        help="path to output dataset: /coc/flash7/datasets/oboov2/<ds_name>.hdf5",
    )

    parser.add_argument(
        "--prestack",
        action="store_true"
    )
    
    # Audio processing arguments
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
        "--enable_audio",
        action="store_true",
        help="Enable audio processing"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )

    args = parser.parse_args()

    main(args)