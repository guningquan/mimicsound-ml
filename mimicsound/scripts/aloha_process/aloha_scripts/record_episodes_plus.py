import os,sys
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import cv2
import queue
import pyaudio
import wave

from constants import DT, START_ARM_POSE, TASK_CONFIGS, DT
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from sleep_plus import sleep_all_robots,shut_down_all_robots
from pynput.keyboard import Key, Listener
import IPython
import threading
e = IPython.embed

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


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, dataset_name):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors

    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)

    def set_operating_modes(robot, arm_mode, gripper_mode):
        robot.dxl.robot_set_operating_modes("group", "arm", arm_mode)
        robot.dxl.robot_set_operating_modes("single", "gripper", gripper_mode)
    def configure_robots(puppet_bot_left, master_bot_left, puppet_bot_right, master_bot_right):
        threads = [
            threading.Thread(target=set_operating_modes, args=(puppet_bot_left, "position", "current_based_position")),
            threading.Thread(target=set_operating_modes, args=(master_bot_left, "position", "position")),
            threading.Thread(target=set_operating_modes, args=(puppet_bot_right, "position", "current_based_position")),
            threading.Thread(target=set_operating_modes, args=(master_bot_right, "position", "position"))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    configure_robots(puppet_bot_left, master_bot_left, puppet_bot_right, master_bot_right)

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start collecting the {dataset_name}.')
    close_thresh = -0.3 #-1.4
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    print("*"*100)
    print(f'Dataset name: {dataset_name}')
    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # --------------------- Audio recording setup (before robot operations) ---------------------
    audio, audio_stream, audio_device_index = setup_audio_device()
    audio_queue = queue.Queue()
    audio_recording = True
    audio_chunks = [[] for _ in range(max_timesteps)]  # Pre-allocate audio storage
    
    def record_audio():
        """Background thread: continuously capture audio data and store in audio_queue"""
        while audio_recording:
            try:
                audio_block = audio_stream.read(AUDIO_BUFFER_SIZE, exception_on_overflow=False)
                audio_queue.put(audio_block)
            except Exception as e:
                print(f"Audio recording error: {e}")
                break
    
    # Start audio recording thread if audio is enabled
    audio_recording_thread = None
    if AUDIO_ENABLED and audio is not None:
        audio_recording_thread = threading.Thread(target=record_audio, daemon=True)
        audio_recording_thread.start()
        print("Audio recording started...")
    # --------------------------------------------------------------------------------------

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, dataset_name)

    # --------------------- Early-stop and mark setup (press E/e to end, M/m to mark) ---------------------
    stop_event = threading.Event()  # set when user presses E/e
    mark_request_queue = queue.Queue()  # queue to store mark requests
    mark_timesteps = []  # store actual mark timesteps
    last_mark_timestep = -1000  # track last mark timestep to enforce minimum distance
    MIN_MARK_DISTANCE = 100  # minimum distance between marks

    def _on_press_episode(key):
        """Keyboard callback during episode recording: press E/e to stop early, M/m to mark."""
        try:
            if key.char in ('e', 'E'):
                stop_event.set()
            elif key.char in ('m', 'M'):
                # Put mark request in queue
                mark_request_queue.put(time.time())
        except AttributeError:
            # Ignore non-char keys (e.g., function keys)
            pass

    kb_listener = Listener(on_press=_on_press_episode)
    kb_listener.start()
    print("Recording... Press 'E' to end this episode early, 'M' to mark timestep.")
    # --------------------------------------------------------------------------------------
    # Data collection (support early stop)
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    time0 = time.time()

    # Clear accumulated audio data before starting data collection
    if AUDIO_ENABLED and audio is not None:
        while not audio_queue.empty():
            audio_queue.get()
        print("Cleared pre-accumulated audio data before starting data collection")

    steps_recorded = 0  # actual number of recorded steps
    try:
        for t in tqdm(range(max_timesteps)):

            if t ==0 and AUDIO_ENABLED and audio is not None:
                while not audio_queue.empty():
                    audio_queue.get()
                print("Cleared pre-accumulated audio data before starting data collection at timestep 0")

            # Check early-stop signal
            if stop_event.is_set():
                print("Early stop requested (E). Finishing this episode gracefully...")
                break

            t0 = time.time()
            action = get_action(master_bot_left, master_bot_right)
            t1 = time.time()
            ts = env.step(action)
            t2 = time.time()

            timesteps.append(ts)
            actions.append(action)
            actual_dt_history.append([t0, t1, t2])
            steps_recorded += 1

            # Check for mark marking requests
            try:
                while not mark_request_queue.empty():
                    mark_request_time = mark_request_queue.get_nowait()
                    # Check if enough time has passed since last mark
                    if t - last_mark_timestep >= MIN_MARK_DISTANCE:
                        mark_timesteps.append(t)
                        last_mark_timestep = t
                        print(f"🎯 Mark marked at timestep {t}")
                    else:
                        remaining_distance = MIN_MARK_DISTANCE - (t - last_mark_timestep)
                        print(f"⚠️ Mark request ignored. Need {remaining_distance} more timesteps between marks.")
            except queue.Empty:
                pass

            # Collect audio data for this timestep
            if AUDIO_ENABLED and audio is not None:
                chunks_collected = 0
                while not audio_queue.empty():
                    audio_chunks[t].append(audio_queue.get())
                    chunks_collected += 1
                if chunks_collected > 0 and t < 5:  # Debug info for first 5 timesteps
                    print(f"Timestep {t}: collected {chunks_collected} audio chunks")

            # Maintain target DT
            time.sleep(max(0, DT - (time.time() - t0)))
    finally:
        # Always stop the keyboard listener to avoid leaking the thread
        kb_listener.stop()
        
        # Stop audio recording
        if AUDIO_ENABLED and audio is not None:
            audio_recording = False
            if audio_recording_thread is not None:
                audio_recording_thread.join()
            audio_stream.stop_stream()
            audio_stream.close()
            audio.terminate()
            print("Audio recording stopped.")

    # Protect against zero-length episodes
    if steps_recorded == 0:
        print("No steps recorded (stopped immediately). Skipping save and re-running...")
        return False

    print(f'Avg fps: {steps_recorded / (time.time() - time0)}')
    if AUDIO_ENABLED and audio is not None:
        print(f"Audio chunks collected: {len([chunk for chunk in audio_chunks if chunk])}")
    
    # Print mark statistics
    if mark_timesteps:
        print(f"🎯 Total marks marked: {len(mark_timesteps)} at timesteps: {mark_timesteps}")
    else:
        print("🎯 No marks were marked during this episode")

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    # Open puppet grippers
    env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 30:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 30, re-collecting... \n\n\n\n')
        return False

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    
    action                  (14,)         'float64'
    base_action             (2,)          'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        # '/base_action': [],
        # '/base_action_t265': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
    
    # Add mark information (handled separately in save_hdf5, so not needed in data_dict)
    # data_dict['/mark_timesteps'] = mark_timesteps


    # while actions:
    #     action = actions.pop(0)
    #     ts = timesteps.pop(0)
    #     data_dict['/observations/qpos'].append(ts.observation['qpos'])
    #     data_dict['/observations/qvel'].append(ts.observation['qvel'])
    #     data_dict['/observations/effort'].append(ts.observation['effort'])
    #     data_dict['/action'].append(action)
    #     for cam_name in camera_names:
    #         data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # Use the actual number of steps recorded to drain buffers deterministically
    for _ in range(steps_recorded):
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)

        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])



    COMPRESS = True 

    if not COMPRESS:
        padded_size = 0
        compressed_len = np.array([])

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()

    def save_hdf5(dataset_path, camera_names, num_steps, padded_size, COMPRESS, data_dict, compressed_len, audio_chunks=None, mark_timesteps=None):
        """Save HDF5 data in a background thread using the *actual* number of steps."""
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')

            # Image datasets sized to the actual number of steps
            for cam_name in camera_names:
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (num_steps, padded_size), dtype='uint8',
                                             chunks=(1, padded_size))
                else:
                    _ = image.create_dataset(cam_name, (num_steps, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3))

            # State/action datasets use actual step count
            _ = obs.create_dataset('qpos', (num_steps, 14))
            _ = obs.create_dataset('qvel', (num_steps, 14))
            _ = obs.create_dataset('effort', (num_steps, 14))
            _ = root.create_dataset('action', (num_steps, 14))
            # _ = root.create_dataset('base_action', (num_steps, 2))

            for name, array in data_dict.items():
                # print(f"Saving {name}")
                root[name][...] = array

            # Save mark timesteps if any
            if mark_timesteps and len(mark_timesteps) > 0:
                _ = root.create_dataset('mark_timesteps', data=np.array(mark_timesteps, dtype=np.int32))
                root.attrs['num_marks'] = len(mark_timesteps)
                print(f"Saved {len(mark_timesteps)} mark timesteps: {mark_timesteps}")
            else:
                root.attrs['num_marks'] = 0


            if AUDIO_ENABLED and audio_chunks is not None:
                max_substeps = 2048  # Maximum audio samples per timestep
                audio_chunks_np = np.zeros((num_steps, max_substeps), dtype=np.int16)
                
                for t in range(num_steps):
                    if t < len(audio_chunks) and audio_chunks[t]:
                        # Combine all audio chunks for this timestep
                        audio_data = np.frombuffer(b"".join(audio_chunks[t]), dtype=np.int16)
                        if len(audio_data) > max_substeps - 1:
                            raise ValueError(
                        f"Time step {t} has too many audio samples ({len(audio_data)} > {max_substeps})! Please check the data.")
                        # print("len(audio_data) for timestep", t, "is", len(audio_data))
                        audio_chunks_np[t, :len(audio_data)] = audio_data
                        audio_chunks_np[t, -1] = len(audio_data)  # Store actual length in last element
                root.create_dataset("audio", data=audio_chunks_np, dtype=np.int16)
                root.attrs['audio_sampling_rate'] = AUDIO_SAMPLING_RATE
                root.attrs['audio_channels'] = AUDIO_CHANNELS
                print(f"Audio data saved: {num_steps} timesteps, max {max_substeps} samples per timestep")


            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(camera_names), num_steps))
                root['/compress_len'][...] = compressed_len

    # 启动后台线程

    audio_data_to_save = audio_chunks if AUDIO_ENABLED and audio is not None else None
    thread = threading.Thread(target=save_hdf5, args=(dataset_path, camera_names, steps_recorded, padded_size, COMPRESS, data_dict, compressed_len, audio_data_to_save, mark_timesteps))

    thread.start()

    # print("HDF5 data storage has started, and the main thread continues executing other tasks...")

    # Wait for the HDF5 thread to complete at some point
    thread.join()
    # print("HDF5 data storage is complete, and the main thread continues executing subsequent code.")

    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    # if not os.path.isdir(dataset_dir):
    #     os.makedirs(dataset_dir)
    # instruction_path = os.path.join(dataset_dir, "instruction.txt")
    # if not os.path.exists(instruction_path):
    #     # Ask for input if file does not exist
    #     instruction = input("Please enter the instruction for this dataset: ")
    #     with open(instruction_path, "w") as f:
    #         f.write(instruction)
    #     print("Instruction saved to instruction.txt")
    # else:
    #     # print("instruction.txt already exists. Skipping input.")
    #     pass


    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')


    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
#     parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
#     main(vars(parser.parse_args())) # TODO
    # debug()
def on_press(key):
    global user_input
    try:
        # 处理键盘字符键
        # if key.char == 'c' or key.char == 'C':
        #     user_input = 'C'
        if key.char == 'c' or key.char == 'C':
            user_input = 'C'
        elif key.char == 'r' or key.char == 'R':
            user_input = 'R'
        elif key.char == 'q' or key.char == 'Q':
            user_input = 'Q'
    except AttributeError:
        # 处理其他键类型，例如功能键等
        pass

def listen_for_key():
    # 监听键盘输入直到获取有效输入
    with Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--start_idx', action='store', type=int, help='Start index.', required=True) # @gnq
    parser.add_argument('--end_idx', action='store', type=int, help='Start index.', required=True)
    # main(vars(parser.parse_args()))
    # debug()

    args = vars(parser.parse_args())

    current_episode = args['start_idx']

    while current_episode <= args['end_idx']:
        args['episode_idx'] = current_episode

        # success = print("eposid", current_episode)
        success = main(args)

        current_episode += 1

        print("*" * 100)
        print(f"The {current_episode-1} episode finished.")
        print("Press 'C' to continue to the next episode, 'R' to repeat this episode, 'Q' to quit:")

        while True:
            user_input = None
            listener = Listener(on_press=on_press)
            listener.start()
            while user_input is None:
                pass
            listener.stop()
            if user_input == 'C':   # left button
                sleep_all_robots()
                # shut_down_all_robots()
                break
            elif user_input == 'R':     # middle button
                sleep_all_robots()
                # shut_down_all_robots()
                current_episode -= 1
                break  # repetition
            elif user_input == 'Q':     # right button
                # sleep_all_robots()
                shut_down_all_robots()
                # delete the quit data
                task_config = TASK_CONFIGS[args['task_name']]
                dataset_dir = task_config['dataset_dir']
                dataset_name = f'episode_{current_episode - 1}.hdf5'
                dataset_path = os.path.join(dataset_dir, dataset_name)
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                    print(f"File {dataset_path} has been deleted successfully.")
                else:
                    print(f"File {dataset_path} does not exist.")
                print(f"The data is saved until {current_episode - 2} episode.")
                sys.exit("Quitting the process.")
            else:
                print("Invalid input, please try again.")

        if current_episode > args['end_idx']:
            shut_down_all_robots()
            print("Completed all episodes.")
            print(f"The data is saved to {current_episode - 1} episode.")
            break