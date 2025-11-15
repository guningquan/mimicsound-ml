# MimicSonic: Learning Bimanual Manipulation from Audio-Visual Human Videos

This repository contains the human-robot audio–vision data collection, processing, and training code for MimicSonic.

---

## 📂 Repo Structure
- **`mimicsonic/scripts/calibration_camera`**: Calibrates the extrinsic pose between the top-view camera and both robot arms.

- **`mimicsonic/scripts/aloha_process`**: Collects robot teleoperation datasets-including images, audio, and robot joint values-and processes raw ALOHA-style data into a RoboMimic-style HDF5 file compatible with training.

- **`mimicsonic/scripts/human_process`**: Collects human manipulation datasets and processes human vision-audio data into a RoboMimic-style HDF5 file compatible with training.

- **`mimicsonic/algo`**: Contains algorithm implementations for MimicSonic.

- **`mimicsonic/configs`**: Provides training configurations for each task.

- **`mimicsonic/scripts/pl_train.py`**: Main training script powered by PyTorch Lightning (with DDP enabled).


## 🏗️ Quick Start Guide

## 🔧 Hardware Setting

1. Our robotic platform is built upon **ALOHA**. Please set up the four robots and cameras according to the original [ALOHA](https://github.com/tonyzhaozh/aloha) instructions. For the top-view camera, we use an **Intel RealSense RGB-D** camera.
2. We strongly recommend that the reader use the  [Foot Pedal Switch](https://www.amazon.co.jp/-/en/gp/product/B07D14Y2NX/ref=ox_sc_act_title_1?smid=A35GGB9A6044W2&psc=1) to facilitate dataset collection.
2. A USB microphone should be connected for receiving human instructions. We use the FIFINE K053 USB Lavalier Lapel Microphone, available at this [link](https://www.amazon.co.jp/-/en/gp/product/B077Y974JF/ref=ox_sc_act_title_1?smid=A17LS08GT0UYE7&psc=1). 
3. You can check your microphone by running:
   ```
   cd mimicsonic/scripts/aloha_process
   python aloha_scripts/audio_mic_test.py # Changing TARGET_DEVICE_NAME to test different mic.
   ```
   This will generate an audio file named "audio.wav".


## 🛠️ Environment Installation
### MimicSonic Installation

```
git clone repo
cd mimicsonic-ml
conda env create -f environment.yaml
source activate mimicsonic
pip install -e external/robomimic
pip install -e .
conda install -n mimicsonic -c conda-forge pyaudio portaudio av==10.0.0
python external/robomimic/robomimic/scripts/setup_macros.py
```

Then go to  `external/robomimic/robomimic/macros_private.py` and manually add your wandb username. Make sure you have ran `wandb login` too.


### SAM Installation
Processing hand and robot data relies on [SAM](https://github.com/facebookresearch/segment-anything-2) to generate masks for the hand and robot.

```
cd outside of mimicsonic-ml
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints && \
./download_ckpts.sh && \
mv sam2_hiera_tiny.pt /path/to/mimicsonic/resources/sam2_hiera_tiny.pt
```

## 📑 Dataset Collection

### Robot Dataset
#### Robotic System Calibration

To train MimicSonic on your own data you must provide the hand-eye-calibration extrinsics matrix inside [``mimicsonic/utils/mimicsonicUtils``](./mimicsonic/utils/mimicsonicUtils.py)
- Print a large april tag and tape it to the wrist camera mount
- Collect calibration data for each arm one at a time.  Move the arm in many directions for best results.  This will generate hdf5. We use the left arm as an example.

```
python mimicsonic/scripts/aloha_process/aloha_scripts/record_episodes_plus.py --task_name CALIBRATE_LEFT --start_idx 0 --end_idx 0 

python mimicsonic/scripts/calibrate_camera/aloha_to_robomimic_calibrate.py  --dataset /Path/CALIBRATE_LEFT  --arm left --out /Path/calibration_left.hdf5  --data-type robot

python mimicsonic/scripts/calibrate_camera/calibrate_realsense.py --h5py-path /Path/calibration_left.hdf5

```
Paste this matrix into [``mimicsonic/utils/mimicsonicUtils``](./mimicsonic/utils/mimicsonicUtils.py) for the appropriate arm.

#### Robot Data Collection and Processing for Training
**Collect Robot Demos**
Run the following command, and use M to mark key frames. Press E to end the current teleoperation once the task is completed.
```
 python mimicsonic/scripts/aloha_process/aloha_scripts/record_episodes_plus.py  --task_name TASK_NAME  --start_idx 0 --end_idx 10
```
We utilize human teleoperation to collect demonstration data. During each task's data collection, information is recorded at each timestep within an episode. Specifically, each timestep includes the current robot joint values, images from the active cameras (top-view, right arm, left arm), and the corresponding audio segment captured at that timestep, which are compressed and saved as HDF5 files. The structure of the dataset is illustrated in the following structure tree:
```
<dataset_root>/
└── <task_name>/                # e.g., alarm_shutting, stapler_checking, etc.,
    ├── episode_0/
    ├── ...
    ├── episode_18/
    │   ├── timestep_0/
    │   ├── ...    
    │   ├── timestep_t/
    │   │   ├── robot_joint_value         # Joint positions
    │   │   ├── camera/                   # Camera images (number of cameras is configurable)
    │   │   │   ├── rgb_cam_top
    │   │   │   ├── rgb_cam_right_arm
    │   │   │   ├── rgb_cam_left_arm
    │   │   ├── audio_current_recorded    # Audio segment for this timestep
    │   │
    │   ├── timestep_t+1/
    │   ├── ...
    ├── episode_19/
    ├── ...
```
This timestep-aligned data collection method can help address synchronization issues in multimodal information, especially in long-horizon tasks. However, because different timesteps may have inconsistent wall-clock durations, the length of audio_current_recorded may vary. In our code, for each timestep, we allocate a fixed-size array large enough to store the corresponding audio segment. The audio segment data for each timestep is stored sequentially from the beginning of the array, with the last element used to record the actual length of the audio data for that timestep.

For each timestep, the audio segment data is stored in a fixed-size array as follows:

```
+----------------+----------------+-----+----------------+-----+------------------+
| sample[0]      | sample[1]      | ... | sample[N-1]    | ... | length_of_audio  |
+----------------+----------------+-----+----------------+-----+------------------+
      ↑                ↑                      ↑                      ↑
  Audio sample 0   Audio sample 1      Last audio sample      Number of valid samples

```
This design facilitates efficient retrieval of the audio data for each timestep and allows for flexible composition of fixed-length audio segments during subsequent training. Although HDF5's variable-length storage was considered, it was found that this approach could easily lead to out-of-memory errors during training.

**Process Robot Demos**

To process the demos we've recorded we run.  Here's an example command
```
python mimicsonic/scripts/aloha_process/aloha_to_robomimic.py \
    --dataset /Path/AlohaData/TASK  \
    --arm both  \
    --out /mimicsonic/TASK.hdf5 \
    --extrinsics rsOct10 \
    --prestack \
    --enable_audio \
    --audio_length 4.0 \
    --target_sampling_rate 16000  \
    --val_ratio 0.2
```


### Human Dataset
**Human Data Collection**
Using the commend to collect human manipulation. 
'b': Start recording,'e': Stop recording and save, 'm': Mark timestep (during 
```python mimicsonic/scripts/human_process/record_hand_rs_plus.py --output_dir /Path/HumanData/TASK  --start_idx 0 --end_idx 40```

**Human Data Procesing**
1. Compute the speed ratio between human and robot. This will output the analyzed speed ratios for different phases of the task and will be used for the subsequent human data generation.
```
python mimicsonic/scripts/analyze_robot_human_speed.py \
        --human_dir /PATH/Human \
        --robot_dir /PATH/Robot \
        --expected_segs 3 \
        --plot
```
2. Convert the human data into robomimic data format:
```
python mimicsonic/scripts/realsense_process/realsense_to_robomimic.py \
    --input_dir /Path/HumanData/TASK \
    --output /Path/HumanMimicsonic/TASK.hdf5 \
    --hand bimanual \
    --enable_audio \
    --audio_length 4.0 \
    --target_sampling_rate 16000 \
    --split \
    --enable_dynamic_speed \
    --seg_speed_ratios x x x \  # The values from analyze_robot_human_speed.py
    --val_ratio 0.5 \
    --augment /recorded/robot_audio_noise.wav
```

## 🧠 **Policy Training**  
1. Configure your settings in mimic/configs/mimicsonic.json.
2. Traing your policy:
```
python  mimicsonic/scripts/pl_train.py \
    --config configs/mimicsonic.json
    --dataset /Path/TASK_robot.hdf5 \
    --dataset_2 /Path/TASK_human.hdf5 \
```

## 📡 **Policy Deployment**

Deploy the trained policy on your robotic platform.

```
python mimicsonic/scripts/eval_real.py  \
    --eval-path  /Path/ckpt
```
