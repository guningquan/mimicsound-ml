"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    bddl_file (str): if provided, the task's goal is specified as the symbolic goal in the bddl file (several symbolic predicates connected with AND / OR)

    video_prompt (str): if provided, a task video prompt is loaded and used in the evaluation rollouts

    debug (bool): set this flag to run a quick training run for debugging purposes
"""

import argparse
import numpy as np
import time
import os
import queue
import threading

import torch
import robomimic.utils.obs_utils as ObsUtils
from torchvision.utils import save_image
import cv2

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("Warning: pyaudio not available, audio recording will be disabled")
# Removed ROS2 dependencies - using act-plus-plus instead
# from interbotix_common_modules.common_robot.robot import (
#     create_interbotix_global_node,
#     robot_shutdown,
#     robot_startup,
# )

# from aloha.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE


from mimicsound.utils.mimicsoundUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    general_unnorm,
    miniviewer,
    nds,
    REALSENSE_INTRINSICS,
    EXTRINSICS,
    ee_pose_to_cam_frame,
    AlohaFK,
)
import torchvision


from mimicsound.configs import config_factory
from mimicsound.pl_utils.pl_model import ModelWrapper
import datetime

# from aloha.robot_utils import move_grippers, move_arms  # requires aloha
# from aloha.real_env import make_real_env  # requires aloha

# Add act-plus-plus dependencies
import sys
import os
# 添加 aloha_scripts 目录到路径，以便相对导入可以工作
aloha_scripts_path = os.path.join(os.path.dirname(__file__), '..', 'aloha_process', 'aloha_scripts')
aloha_scripts_path = os.path.abspath(aloha_scripts_path)
sys.path.insert(0, aloha_scripts_path)

from robot_utils import move_grippers, move_arms
from real_env import make_real_env
from constants import DT, PUPPET_GRIPPER_JOINT_OPEN as FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE

from mimicsound.scripts.evaluation.real_utils import *
import matplotlib.pyplot as plt
from mimicsound.algo.act import ACT
from mimicsound.scripts.masking.utils import SAM

import pickle


# For debugging
# sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


CURR_INTRINSICS = REALSENSE_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["rsOct10R"]
# NORM_STATS = to_torch(NORM_STATS, torch.device("cuda"))
CAM_KEY = "front_img_1"   # front_img_1_line, front_img_1 # @gnq
TEMPORAL_AGG = True # False
query_frequency = 2

# Audio configuration
ENABLE_AUDIO = True  # Set to False to disable audio recording
AUDIO_SAMPLING_RATE = 48000  # Audio sampling rate in Hz
AUDIO_LENGTH = 2.0  # Audio length in seconds (2 seconds)
AUDIO_CHUNK = 256  # Audio chunk size for recording
AUDIO_TARGET_SAMPLING_RATE = 16000  # Target sampling rate for model input
AUDIO_TARGET_DEVICE_NAME = "USB PnP Audio Device"  # Audio device name to search for 


class TemporalAgg:
    def __init__(self, query_frequency=25):
        self.recent_actions = []
        self.action_dim = None  # Will be set dynamically
        self.query_frequency = query_frequency  # 查询频率，用于时间对齐
    
    def add_action(self, action):
        """
            actions: (100, action_dim) tensor
        """
        if self.action_dim is None:
            self.action_dim = action.shape[1]  # Set action dimension from first action
        self.recent_actions.append(action)
        if len(self.recent_actions) > 4:
            del self.recent_actions[0]

    def smoothed_action(self):
        """
            returns smooth action (100, action_dim)
        """
        if self.action_dim is None:
            raise ValueError("No actions added yet, cannot determine action dimension")
            
        mask = []
        count = 0

        shifted_actions = []
        # breakpoint()

        for ac in self.recent_actions[::-1]:
            basic_mask = np.zeros(100)
            basic_mask[:100-count] = 1
            mask.append(basic_mask)
            shifted_ac = ac[count:]
            shifted_ac = np.concatenate([shifted_ac, np.zeros((count, self.action_dim))], axis=0)
            shifted_actions.append(shifted_ac)
            count += self.query_frequency

        mask = mask[::-1]
        mask = ~(np.array(mask).astype(bool))
        recent_actions = shifted_actions[::-1]
        recent_actions = np.array(recent_actions)
        # breakpoint()
        mask = np.repeat(mask[:, :, None], self.action_dim, axis=2)
        smoothed_action = np.ma.array(recent_actions, mask=mask).mean(axis=0)

        # PLOT_JOINT = 0
        # for i in range(recent_actions.shape[0]):
        #     plt.plot(recent_actions[i, :, PLOT_JOINT], label=f"index{i}")
        # plt.plot(smoothed_action[:, PLOT_JOINT], label="smooth")
        # plt.legend()
        # plt.savefig("smoothing.png")
        # plt.close()
        # breakpoint()

        return smoothed_action

def eval_real(model, env, rollout_dir, norm_stats, arm):
    print("arm is", arm) # @gnq

    device = torch.device("cuda")

    aloha_fk = AlohaFK()
    sam = SAM()

    # Initialize audio if enabled
    audio_device_index = None
    if ENABLE_AUDIO and pyaudio is not None:
        print("Initializing audio recording...")
        audio = pyaudio.PyAudio()
        # Find audio device
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if AUDIO_TARGET_DEVICE_NAME in device_info["name"]:
                audio_device_index = i
                print(f"Found audio device: {device_info['name']}, Index: {audio_device_index}")
                break
        audio.terminate()
        if audio_device_index is None:
            print(f"Warning: Audio device '{AUDIO_TARGET_DEVICE_NAME}' not found. Audio recording disabled.")
            audio_device_index = None

    # max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    qpos_t, actions_t = [], []
    num_rollouts = 50

    for rollout_id in range(num_rollouts):
        print(f"Starting rollout: {rollout_id}")
        input("Enter to start the rollout")
        if TEMPORAL_AGG:
            TA = TemporalAgg(query_frequency=query_frequency)

        # Initialize audio recording for this rollout
        audio_queue = None
        audio_stream = None
        audio = None
        recording = False
        recording_thread = None
        
        if ENABLE_AUDIO and pyaudio is not None and audio_device_index is not None:
            audio = pyaudio.PyAudio()
            audio_stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_SAMPLING_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK,
                input_device_index=audio_device_index
            )
            audio_queue = queue.Queue()
            recording = True
            
            def record_audio():
                """Background thread: continuously record audio data and store in audio_queue"""
                while recording:
                    try:
                        audio_block = audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                        audio_queue.put(audio_block)
                    except Exception as e:
                        print(f"Audio recording error: {e}")
                        break
            
            recording_thread = threading.Thread(target=record_audio, daemon=True)
            recording_thread.start()
            print("Audio recording started...")

        ts = env.reset()

        t0 = time.time()
        with torch.inference_mode():
            rollout_images = []
            # Calculate max audio length in samples at original sampling rate
            max_audio_length_original = int(AUDIO_SAMPLING_RATE * AUDIO_LENGTH)
            # Target length after resampling
            max_audio_length_target = int(AUDIO_TARGET_SAMPLING_RATE * AUDIO_LENGTH)
            
            for t in range(1000):
                time.sleep(max(0, DT*2 - (time.time() - t0)))
                # print(f"DT: {time.time() - t0}")
                t0 = time.time()

                obs = ts.observation
                # plt.imsave(os.path.join(rollout_dir, f"viz{t}.png"), obs["images"]["cam_high"])
                # plt.imsave(os.path.join(rollout_dir, f"wrist{t}.png"), obs["images"]["cam_right_wrist"])

                qpos = np.array(obs["qpos"])
                qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
                inference_t = time.time()


                ### query policy
                if t % query_frequency == 0:
                    # Process audio data if available
                    audio_chunk = None
                    if ENABLE_AUDIO and audio_queue is not None:
                        # Collect all available audio chunks from queue
                        Audio_Chunks_Original = []
                        while not audio_queue.empty():
                            Audio_Chunks_Original.append(audio_queue.get())
                        
                        if Audio_Chunks_Original:
                            # Convert audio chunks to numpy array
                            Audio_Chunks = np.frombuffer(b"".join(Audio_Chunks_Original), dtype=np.int16)
                            
                            # Process audio: truncate or pad to target length (at original sampling rate)
                            if len(Audio_Chunks) > max_audio_length_original:
                                audio_chunk = Audio_Chunks[-max_audio_length_original:]
                            elif len(Audio_Chunks) < max_audio_length_original:
                                pad_length = max_audio_length_original - len(Audio_Chunks)
                                zero_padding = np.zeros((pad_length,), dtype=Audio_Chunks.dtype)
                                audio_chunk = np.concatenate((zero_padding, Audio_Chunks))
                            else:
                                audio_chunk = Audio_Chunks
                            
                            # Resample from original rate (48kHz) to target rate (16kHz) if needed
                            if AUDIO_SAMPLING_RATE != AUDIO_TARGET_SAMPLING_RATE:
                                from scipy import signal
                                audio_chunk = signal.resample(audio_chunk, max_audio_length_target)
                            
                            # Normalize to [-1, 1] range
                            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                            
                            # Convert to torch tensor
                            # Add time dimension: (1, audio_length) -> (1, 1, audio_length)
                            # This matches the expected shape for process_batch_for_training
                            audio_chunk = torch.tensor(audio_chunk, device=device).unsqueeze(0).unsqueeze(0)
                        else:
                            # No audio data available, create zero audio with time dimension
                            audio_chunk = torch.zeros((1, 1, max_audio_length_target), device=device, dtype=torch.float32)

                    # right wrist data
                    data = {
                        "obs": {
                            "right_wrist_img": (
                                torch.from_numpy(obs["images"]["cam_right_wrist"][None, None, :])
                            ).to(torch.uint8),
                            "pad_mask": torch.ones((1, 100, 1)).to(device).bool(),
                            "joint_positions": qpos[..., 7:].reshape((1, 1, -1)),
                        },
                        "type": torch.tensor([0]),
                    }
                    
                    # Add audio to data if available
                    # Audio shape should be (B, T, audio_length) to match process_batch_for_training expectations
                    if audio_chunk is not None:
                        data["obs"]["audio"] = audio_chunk

                    # add regular or line overlay top camera
                    if CAM_KEY == "front_img_1":
                        data["obs"][CAM_KEY] = torch.from_numpy(
                            obs["images"]["cam_high"][None, None, :]
                        ).to(torch.uint8)

                    if arm == "right":
                        data["obs"]["joint_positions"] =  qpos[..., 7:].reshape((1, 1, -1))
                        
                        if CAM_KEY == "front_img_1_line":
                            _, line_image = sam.get_robot_mask_line_batched_from_qpos(obs["images"]["cam_high"][None, :], qpos, EXTRINSICS["rsOct10"], REALSENSE_INTRINSICS, arm=arm)
                            line_image = line_image[0]
                            data["obs"][CAM_KEY] = torch.from_numpy(
                                line_image[None, None, :]
                            ).to(torch.uint8)

                        # postprocess_batch
                        input_batch = model.process_batch_for_training(
                            data, "actions_joints_act"
                        )

                        input_batch["obs"]["right_wrist_img"] = input_batch["obs"]["right_wrist_img"].permute(0, 3, 1, 2)
                        input_batch["obs"]["right_wrist_img"] /= 255.0
                    
                    elif arm == "both":
                        data["obs"]["left_wrist_img"] = torch.from_numpy(obs["images"]["cam_left_wrist"][None, None, :]).to(torch.uint8)
                        data["obs"]["joint_positions"] = qpos[..., :].reshape((1, 1, -1))


                        if CAM_KEY == "front_img_1_line":
                            _, line_image = sam.get_robot_mask_line_batched_from_qpos(obs["images"]["cam_high"][None, :], qpos, EXTRINSICS["rsOct10"], REALSENSE_INTRINSICS, arm=arm)
                            line_image = line_image[0]
                            data["obs"][CAM_KEY] = torch.from_numpy(
                                line_image[None, None, :]
                            ).to(torch.uint8)

                        # postprocess_batch
                        input_batch = model.process_batch_for_training(
                            data, "actions_joints_act"
                        )

                        # right
                        input_batch["obs"]["right_wrist_img"] = input_batch["obs"]["right_wrist_img"].permute(0, 3, 1, 2)/255.0

                        # left
                        input_batch["obs"]["left_wrist_img"] = input_batch["obs"]["left_wrist_img"].permute(0, 3, 1, 2)/255.0

                    # breakpoint()
                    input_batch["obs"][CAM_KEY] = input_batch["obs"][CAM_KEY].permute(0, 3, 1, 2)
                    input_batch["obs"][CAM_KEY] /= 255.0
                    input_batch = ObsUtils.normalize_batch(input_batch, normalization_stats=norm_stats, normalize_actions=False)
                    
                    # Prepare audio for model forward if available
                    # Note: The model's forward_eval should handle audio if it's in input_batch
                    # If the model expects audio as a separate argument, modify this accordingly
                    info = model.forward_eval(input_batch, unnorm_stats=norm_stats)

                    all_actions = info["actions_joints_act"].cpu().numpy()

                    if TEMPORAL_AGG:
                        TA.add_action(all_actions[0])
                        all_actions = TA.smoothed_action()[None, :]


                    if rollout_dir:
                        # Draw Actions
                        im = data["obs"][CAM_KEY][0, 0].cpu().numpy()
                        pred_values = info["actions_joints_act"][0].cpu().numpy()

                        if "joints" in model.ac_key:
                            pred_values_drawable = aloha_fk.fk(pred_values[:, :6])
                            pred_values_drawable = ee_pose_to_cam_frame(pred_values_drawable, CURR_EXTRINSICS)
                        else:
                            pred_values_drawable = pred_values


                        pred_values_drawable = cam_frame_to_cam_pixels(
                            pred_values_drawable, CURR_INTRINSICS
                        )

                        im = np.array(im, dtype="uint8")
                        frame = draw_dot_on_frame(
                            im, pred_values_drawable[[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]], show=False, palette="Greens"
                        )


                        # Draw ee_pose
                        ee_pose_input = aloha_fk.fk(qpos[:, 7:13]).to(device)
                        ee_pose_cam_frame = ee_pose_to_cam_frame(
                            ee_pose_input.cpu().numpy(), CURR_EXTRINSICS
                        )[:, None, :]
                        ee_pose_pixels = cam_frame_to_cam_pixels(
                            ee_pose_cam_frame[0], CURR_INTRINSICS
                        )
                        frame = draw_dot_on_frame(
                            frame, ee_pose_pixels, show=False, palette="Set1"
                        )

                        # Save images
                        rollout_images.append(frame)
                        plt.imsave(os.path.join(rollout_dir, f"viz{t}.png"), frame)
                        plt.imsave(os.path.join(rollout_dir, f"wrist_rgb{t}.png"), data["obs"]["right_wrist_img"][0, 0].cpu().numpy())
                    
                    print(f"Inference time: {time.time() - inference_t}")

                raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action[0]
                # action = post_process(raw_action)
                target_qpos = raw_action
                # target_qpos = action

                ### step the environment
                if arm == "right":
                    target_qpos = np.concatenate([np.zeros(7), target_qpos])

                # print("target_qpos is", target_qpos) # @gnq
                # input("Enter to continue the action") # @gnq
                
                ts = env.step(target_qpos)

                # debugging control loop
                qpos_t.append(ts.observation["qpos"])
                actions_t.append(target_qpos)
            
            # Stop audio recording
            if ENABLE_AUDIO and recording_thread is not None:
                recording = False
                recording_thread.join(timeout=1.0)
                if audio_stream is not None:
                    audio_stream.stop_stream()
                    audio_stream.close()
                if audio is not None:
                    audio.terminate()
                print("Audio recording stopped.")

        if rollout_dir:
            qpos_t = np.array(qpos_t)
            actions_t = np.array(actions_t)
            for i in range(7, 14):
                plt.plot(qpos_t[:, i], label=f"qpos joint {i}")
                plt.plot(actions_t[:, i], label=f"ac joint {i}")
                plt.legend()

                plt.savefig(f"/home/rl2-bonjour/EgoPlay/EgoPlay/debug_ims/joint{i}_actions.png", dpi=300)
                plt.close()

                plt.plot(actions_t[:, i] - qpos_t[:, i], label="error joint{i}")
                plt.legend()

                plt.savefig(f"/home/rl2-bonjour/EgoPlay/EgoPlay/debug_ims/joint{i}_error.png", dpi=300)
                plt.close()


        # save_images(rollout_images, viz_dir)
        # write_vid(rollout_images, os.path.join(viz_dir, "video_0.mp4"))
        rollout_images = []

        print("moving robot")
        if arm == "right":
            move_grippers(
                [env.puppet_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN], moving_time=0.5
            )  # open
            move_arms([env.puppet_bot_right], [START_ARM_POSE[:6]], moving_time=1.0)
        elif arm == "both":
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN]*2, moving_time=0.5
            )  # open
            move_arms([env.puppet_bot_left, env.puppet_bot_right], [START_ARM_POSE[:6]]*2, moving_time=1.0)

        time.sleep(12.0)
    return


def main(args):
    """
    Train a model using the algorithm.
    """
    # first set seeds
    np.random.seed(101)
    torch.manual_seed(101)

    # print("\n============= New Training Run with Config =============")
    # print(config)
    # print("")
    # log_dir, ckpt_dir, video_dir, uid = get_exp_dir(config)

    # breakpoint()
    model = ModelWrapper.load_from_checkpoint(args.eval_path, datamodule=None)
    norm_stats = os.path.join(os.path.dirname(os.path.dirname(args.eval_path)), "ds1_norm_stats.pkl")
    norm_stats = open(norm_stats, "rb")
    norm_stats = pickle.load(norm_stats)
    
    # Removed ROS2 node creation - using act-plus-plus direct interface
    # node = create_interbotix_global_node('aloha')
    arm = "both" # @gnq
    if model.model.ac_dim == 14:
        arm = "both"
        env = make_real_env(init_node=True, setup_robots=True)
    elif model.model.ac_dim == 7:
        arm = "right"
        env = make_real_env(init_node=True, setup_robots=True)
    # robot_startup(node)  # Not needed with act-plus-plus
    model.eval()
    rollout_dir = os.path.dirname(os.path.dirname(args.eval_path))
    rollout_dir = os.path.join(rollout_dir, "rollouts")
    if not os.path.exists(rollout_dir):
        os.mkdir(rollout_dir)

    if not args.debug:
        rollout_dir = None

    eval_real(model.model, env, rollout_dir, norm_stats, arm=arm)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="(optional) path to the model to be evaluated",
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    args = parser.parse_args()
    # if "DT" not in args.description:
    #     time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
    #     args.description = time_str
    main(args)
