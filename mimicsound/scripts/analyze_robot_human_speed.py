#!/usr/bin/env python3
"""
Robot vs Human Speed Analysis Script

Analyze speed differences between robot and human data across sub-tasks (segments)
Compare duration of different segments, calculate speed ratios for dynamic speed adjustment

Usage:
python scripts/analyze_robot_human_speed.py --expected_segs 3
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse

class SpeedAnalyzer:
    """Analyze speed differences between robot and human data"""
    
    def __init__(self, human_dir: str, robot_dir: str, expected_segs: int = None):
        """
        Initialize analyzer
        
        Args:
            human_dir: Human data directory path
            robot_dir: Robot data directory path
            expected_segs: Expected number of segments (if None, auto-detect)
        """
        self.human_dir = human_dir
        self.robot_dir = robot_dir
        self.expected_segs = expected_segs
        self.human_data = []
        self.robot_data = []
        
    def load_data(self):
        """Load all data files"""
        print("=== Loading Human Data ===")
        self.human_data = self._load_directory_data(self.human_dir, "human")
        
        print("\n=== Loading Robot Data ===")
        self.robot_data = self._load_directory_data(self.robot_dir, "robot")
        
        # Validate segment consistency
        self._validate_segment_consistency()
    
    def _validate_segment_consistency(self):
        """Validate that all files have the same number of segments"""
        print("\n=== Validating Segment Consistency ===")
        
        # Collect segment counts from all files
        human_seg_counts = []
        robot_seg_counts = []
        
        for data in self.human_data:
            seg_count = len(data['seg_info'])
            human_seg_counts.append(seg_count)
            print(f"Human {data['filename']}: {seg_count} segments")
            
        for data in self.robot_data:
            seg_count = len(data['seg_info'])
            robot_seg_counts.append(seg_count)
            print(f"Robot {data['filename']}: {seg_count} segments")
        
        # Check if all human files have the same segment count
        if len(set(human_seg_counts)) > 1:
            raise ValueError(f"❌ Human data files have inconsistent segment counts: {human_seg_counts}")
        
        # Check if all robot files have the same segment count
        if len(set(robot_seg_counts)) > 1:
            raise ValueError(f"❌ Robot data files have inconsistent segment counts: {robot_seg_counts}")
        
        # Check if human and robot have the same segment count
        if human_seg_counts[0] != robot_seg_counts[0]:
            raise ValueError(f"❌ Human and robot data have different segment counts: Human={human_seg_counts[0]}, Robot={robot_seg_counts[0]}")
        
        detected_segs = human_seg_counts[0]
        print(f"✅ All files have consistent segment count: {detected_segs}")
        
        # Check against expected segment count if provided
        if self.expected_segs is not None:
            if detected_segs != self.expected_segs:
                raise ValueError(f"❌ Expected {self.expected_segs} segments, but detected {detected_segs}")
            print(f"✅ Segment count matches expected value: {self.expected_segs}")
        
        self.actual_segs = detected_segs
        
    def _load_directory_data(self, directory: str, data_type: str) -> List[Dict]:
        """Load all HDF5 files from directory"""
        data_list = []
        
        if not os.path.exists(directory):
            print(f"❌ Directory does not exist: {directory}")
            return data_list
            
        hdf5_files = [f for f in os.listdir(directory) if f.endswith('.hdf5')]
        hdf5_files.sort()
        
        print(f"Found {len(hdf5_files)} HDF5 files: {hdf5_files}")
        
        for filename in hdf5_files:
            filepath = os.path.join(directory, filename)
            try:
                data = self._load_single_file(filepath, data_type)
                if data:
                    data['filename'] = filename
                    data_list.append(data)
                    print(f"✅ Successfully loaded: {filename}")
                else:
                    print(f"❌ Failed to load: {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                
        return data_list
    
    def _load_single_file(self, filepath: str, data_type: str) -> Dict:
        """Load single HDF5 file"""
        with h5py.File(filepath, 'r') as f:
            data = {}
            
            # Check for required data
            if 'mark_timesteps' not in f:
                print(f"⚠️  File {filepath} does not have mark_timesteps")
                return None
                
            data['mark_timesteps'] = f['mark_timesteps'][:]
            
            # Get total frame count
            if 'rgb' in f:
                # Human data
                total_frames = len(f['rgb'])
            elif 'observations' in f and 'qpos' in f['observations']:
                # Robot data
                total_frames = len(f['observations']['qpos'])
            else:
                print(f"⚠️  Cannot determine total frame count: {filepath}")
                return None
                
            data['total_frames'] = total_frames
            
            # Load timestamps
            if 'timestamps' in f:
                data['timestamps'] = f['timestamps'][:]
                data['total_duration'] = data['timestamps'][-1] - data['timestamps'][0]
            else:
                # If no timestamps, estimate using frame count (assume 30fps)
                data['total_duration'] = total_frames / 30.0  # Assume 30fps
                data['timestamps'] = None
                
            # Calculate segment info
            data['seg_info'] = self._calculate_seg_info(data['mark_timesteps'], data['timestamps'], total_frames)
            
            return data
    
    def _calculate_seg_info(self, mark_timesteps: np.ndarray, timestamps: np.ndarray = None, total_frames: int = None) -> List[Dict]:
        """Calculate information for each segment"""
        seg_info = []
        
        # Two mark points create 3 segments
        # Seg 0: 0 -> mark_timesteps[0]
        # Seg 1: mark_timesteps[0] -> mark_timesteps[1]  
        # Seg 2: mark_timesteps[1] -> total_frames
        
        boundaries = [0] + list(mark_timesteps) + [total_frames]
        
        for i in range(len(boundaries) - 1):
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]
            duration_frames = end_frame - start_frame
            
            seg_data = {
                'seg_id': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': duration_frames
            }
            
            # If timestamps available, calculate actual time
            if timestamps is not None and len(timestamps) > end_frame:
                start_time = timestamps[start_frame]
                end_time = timestamps[end_frame - 1]  # end_frame is exclusive
                seg_data['start_time'] = start_time
                seg_data['end_time'] = end_time
                seg_data['duration_seconds'] = end_time - start_time
            else:
                # Use frame count to estimate time (assume 30fps)
                seg_data['duration_seconds'] = duration_frames / 30.0
                
            seg_info.append(seg_data)
            
        return seg_info
    
    def analyze_speeds(self):
        """Analyze speed differences"""
        print("\n=== Speed Analysis ===")
        
        # Calculate average segment durations for human and robot
        human_seg_durations = self._get_average_seg_durations(self.human_data, "Human")
        robot_seg_durations = self._get_average_seg_durations(self.robot_data, "Robot")
        
        # Calculate speed ratios
        speed_ratios = self._calculate_speed_ratios(human_seg_durations, robot_seg_durations)
        
        # Generate report
        self._generate_report(human_seg_durations, robot_seg_durations, speed_ratios)
        
        return speed_ratios
    
    def _get_average_seg_durations(self, data_list: List[Dict], data_type: str) -> List[float]:
        """Calculate average segment durations"""
        print(f"\n--- {data_type} Segment Duration Analysis ---")
        
        # Collect all segment durations
        seg_durations = {}  # {seg_id: [duration1, duration2, ...]}
        
        for data in data_list:
            for seg_info in data['seg_info']:
                seg_id = seg_info['seg_id']
                duration = seg_info['duration_seconds']
                
                if seg_id not in seg_durations:
                    seg_durations[seg_id] = []
                seg_durations[seg_id].append(duration)
                
                print(f"  {data['filename']} - Seg {seg_id}: {duration:.3f}s ({seg_info['duration_frames']} frames)")
        
        # Calculate averages
        avg_durations = []
        for seg_id in sorted(seg_durations.keys()):
            durations = seg_durations[seg_id]
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            avg_durations.append(avg_duration)
            
            print(f"  Seg {seg_id} Average: {avg_duration:.3f}s ± {std_duration:.3f}s (n={len(durations)})")
            
        return avg_durations
    
    def _calculate_speed_ratios(self, human_durations: List[float], robot_durations: List[float]) -> List[float]:
        """Calculate speed ratios (robot/human)"""
        print(f"\n--- Speed Ratio Calculation ---")
        
        speed_ratios = []
        min_segs = min(len(human_durations), len(robot_durations))
        
        for i in range(min_segs):
            human_dur = human_durations[i]
            robot_dur = robot_durations[i]
            
            # Speed ratio = robot time / human time
            # If robot time is longer, robot is slower, ratio > 1
            # If robot time is shorter, robot is faster, ratio < 1
            ratio = robot_dur / human_dur
            speed_ratios.append(ratio)
            
            print(f"  Seg {i}: Human {human_dur:.3f}s, Robot {robot_dur:.3f}s, Ratio {ratio:.3f}")
            
        return speed_ratios
    
    def _generate_report(self, human_durations: List[float], robot_durations: List[float], speed_ratios: List[float]):
        """Generate analysis report"""
        print(f"\n{'='*60}")
        print("📊 ROBOT vs HUMAN SPEED ANALYSIS REPORT")
        print(f"{'='*60}")
        
        print(f"\n📈 Average Segment Duration Comparison:")
        print(f"{'Seg':<4} {'Human(s)':<10} {'Robot(s)':<10} {'Ratio':<8} {'Description'}")
        print("-" * 60)
        
        for i in range(len(speed_ratios)):
            ratio = speed_ratios[i]
            if ratio > 1.5:
                description = "Robot significantly slower"
            elif ratio > 1.1:
                description = "Robot slightly slower"
            elif ratio < 0.9:
                description = "Robot faster"
            elif ratio < 0.7:
                description = "Robot significantly faster"
            else:
                description = "Similar speed"
                
            print(f"{i:<4} {human_durations[i]:<10.3f} {robot_durations[i]:<10.3f} {ratio:<8.3f} {description}")
        
        print(f"\n🎯 Dynamic Speed Adjustment Recommendations:")
        print(f"Recommended seg_speed_ratios parameters: {speed_ratios}")
        
        # Generate command for realsense_to_robomimic.py
        ratios_str = " ".join([f"{r:.3f}" for r in speed_ratios])
        print(f"\n💻 Usage Recommendation:")
        print(f"python mimicsound/scripts/realsense_process/realsense_to_robomimic.py --enable_dynamic_speed --seg_speed_ratios {ratios_str}")
        
        # Save results to files
        self._save_results(human_durations, robot_durations, speed_ratios)
    
    def _save_results(self, human_durations: List[float], robot_durations: List[float], speed_ratios: List[float]):
        """Save analysis results to files"""
        results = {
            'human_avg_durations': human_durations,
            'robot_avg_durations': robot_durations,
            'speed_ratios': speed_ratios,
            'recommended_seg_speed_ratios': speed_ratios,
            'actual_segs': self.actual_segs
        }
        
        # Save as numpy file
        np.save('speed_analysis_results.npy', results)
        
        # Save as CSV (manual write)
        with open('speed_analysis_results.csv', 'w') as f:
            f.write('seg_id,human_duration,robot_duration,speed_ratio\n')
            for i in range(len(speed_ratios)):
                f.write(f'{i},{human_durations[i]:.6f},{robot_durations[i]:.6f},{speed_ratios[i]:.6f}\n')
        
        print(f"\n💾 Results saved to:")
        print(f"  - speed_analysis_results.npy")
        print(f"  - speed_analysis_results.csv")
    
    def plot_comparison(self):
        """Generate comparison plots"""
        if not self.human_data or not self.robot_data:
            print("❌ No data available for plotting")
            return
            
        # Collect data for plotting
        human_segs = []
        robot_segs = []
        
        for data in self.human_data:
            for seg_info in data['seg_info']:
                human_segs.append({
                    'seg_id': seg_info['seg_id'],
                    'duration': seg_info['duration_seconds'],
                    'filename': data['filename']
                })
                
        for data in self.robot_data:
            for seg_info in data['seg_info']:
                robot_segs.append({
                    'seg_id': seg_info['seg_id'],
                    'duration': seg_info['duration_seconds'],
                    'filename': data['filename']
                })
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Robot vs Human Speed Analysis', fontsize=16)
        
        # 1. Duration comparison
        ax1 = axes[0, 0]
        seg_ids = sorted(set([s['seg_id'] for s in human_segs + robot_segs]))
        
        human_means = []
        robot_means = []
        human_stds = []
        robot_stds = []
        
        for seg_id in seg_ids:
            human_durs = [s['duration'] for s in human_segs if s['seg_id'] == seg_id]
            robot_durs = [s['duration'] for s in robot_segs if s['seg_id'] == seg_id]
            
            human_means.append(np.mean(human_durs))
            robot_means.append(np.mean(robot_durs))
            human_stds.append(np.std(human_durs))
            robot_stds.append(np.std(robot_durs))
        
        x = np.arange(len(seg_ids))
        width = 0.35
        
        ax1.bar(x - width/2, human_means, width, yerr=human_stds, label='Human', alpha=0.8)
        ax1.bar(x + width/2, robot_means, width, yerr=robot_stds, label='Robot', alpha=0.8)
        ax1.set_xlabel('Segment ID')
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Average Segment Duration Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Seg {i}' for i in seg_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speed ratios
        ax2 = axes[0, 1]
        speed_ratios = [robot_means[i] / human_means[i] for i in range(len(seg_ids))]
        bars = ax2.bar(seg_ids, speed_ratios, alpha=0.8, color='orange')
        ax2.set_xlabel('Segment ID')
        ax2.set_ylabel('Speed Ratio (Robot/Human)')
        ax2.set_title('Speed Ratio by Segment')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Speed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars, speed_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom')
        
        # 3. Scatter plot: Human vs Robot
        ax3 = axes[1, 0]
        ax3.scatter(human_means, robot_means, alpha=0.7, s=100)
        ax3.plot([0, max(max(human_means), max(robot_means))], 
                [0, max(max(human_means), max(robot_means))], 
                'r--', alpha=0.7, label='Equal Speed Line')
        ax3.set_xlabel('Human Duration (seconds)')
        ax3.set_ylabel('Robot Duration (seconds)')
        ax3.set_title('Human vs Robot Duration Scatter Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add segment labels
        for i, (h, r) in enumerate(zip(human_means, robot_means)):
            ax3.annotate(f'Seg {i}', (h, r), xytext=(5, 5), textcoords='offset points')
        
        # 4. Speed ratio distribution
        ax4 = axes[1, 1]
        ax4.hist(speed_ratios, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Equal Speed')
        ax4.set_xlabel('Speed Ratio')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Speed Ratio Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('speed_analysis_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Plot saved to: speed_analysis_plot.png")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze speed differences between Robot and Human data")
    parser.add_argument(
        "--human_dir",
        type=str,
        default="/path",
        help="Human data directory path"
    )
    parser.add_argument(
        "--robot_dir", 
        type=str,
        default="/path",
        help="Robot data directory path"
    )
    parser.add_argument(
        "--expected_segs",
        type=int,
        default=None,
        help="Expected number of segments (if None, auto-detect)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SpeedAnalyzer(args.human_dir, args.robot_dir, args.expected_segs)
    
    try:
        # Load data
        analyzer.load_data()
        
        if not analyzer.human_data:
            print("❌ No Human data found")
            return
            
        if not analyzer.robot_data:
            print("❌ No Robot data found")
            return
        
        # Analyze speeds
        speed_ratios = analyzer.analyze_speeds()
        
        # Generate plots
        if args.plot:
            analyzer.plot_comparison()
        
        print(f"\n✅ Analysis completed successfully!")
        
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
