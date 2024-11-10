import os
import shutil
import tempfile
import subprocess
import re
import torch
import hashlib
import random
import numpy as np
import mmcv
import pickle
import signal
import sys
from mmengine.config import Config
from mmseg.models.backbones.lpsnet import LPSNet
from mmseg.models.decode_heads import LPSNetHead
import time
import glob
import pandas as pd 
import openpyxl

def measure_latency_ms(model, input_tensor, iterations=100):
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    # Measure latency
    torch.cuda.synchronize()
    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
            timings.append(elapsed)
    latency = sum(timings) / len(timings)
    return latency

class LPSNetExpansion:
    def __init__(self, depth, width, resolution, parent=None):
        self.depth = depth
        self.width = width
        self.resolution = resolution
        self._check_net_param()
        self.parent = parent
        self.latency = self.eval_latency()
        self.miou = -1

        self.delta_depth = ((0, 1, 1, 1, 1), (0, 0, 1, 1, 1), (0, 0, 0, 1, 1))
        self.delta_width = ((4, 8, 16, 32, 32), (0, 8, 16, 32, 32), (0, 0, 16, 32, 32), (0, 0, 0, 32, 32))
        self.delta_resolution = ((1/8, 0, 0), (0, 1/8, 0), (0, 0, 1/8))

    def _check_net_param(self):
        assert len(self.depth) == 5
        assert len(self.width) == 5
        assert len(self.resolution) == 3
        assert all([d > 0 for d in self.depth])
        assert all([w > 0 for w in self.width])
        assert any([r >= 0 for r in self.resolution])

    @property
    def hash(self):
        arch = '{}-{}-{}'.format(self.depth, self.width, sorted(self.resolution, reverse=True))
        sha1 = hashlib.sha1(arch.encode("utf-8"))
        return sha1.hexdigest()

    def eval_latency(self, datashape=(1, 3, 768, 768)):
        model = LPSNet(depth=self.depth, width=self.width, resolution=self.resolution)
        model.eval()
        model.cuda()
        data = torch.rand(datashape).cuda()
        lat = measure_latency_ms(model, data)
        return lat

    def _expand_depth(self, op, steplen=1):
        new_depth = [i + j * steplen for i, j in zip(self.depth, op)]
        arch = LPSNetExpansion(depth=new_depth, width=self.width, resolution=self.resolution, parent=self)
        print('    expand_depth: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.depth, op, steplen, arch.depth, arch.latency))
        return arch

    def _expand_width(self, op, steplen=1):
        new_width = [i + j * steplen for i, j in zip(self.width, op)]
        arch = LPSNetExpansion(depth=self.depth, width=new_width, resolution=self.resolution, parent=self)
        print('    expand_width: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.width, op, steplen, arch.width, arch.latency))
        return arch

    def _expand_resolution(self, op, steplen=1):
        new_resolution = [i + j * steplen for i, j in zip(self.resolution, op)]
        arch = LPSNetExpansion(depth=self.depth, width=self.width, resolution=new_resolution, parent=self)
        print('    expand_resolution: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.resolution, op, steplen, arch.resolution, arch.latency))
        return arch

    def _expand_depth_to_target(self, target):
        arch_save = []
        for op in self.delta_depth:
            steplen = 1
            while True:
                arch = self._expand_depth(op, steplen)
                if arch.latency >= target:
                    break
                steplen += 1
            # Now arch.latency >= target
            # Also compute arch_prev
            arch_prev = self._expand_depth(op, steplen - 1)
            # Choose the architecture closer to the target latency
            if abs(target - arch_prev.latency) < abs(target - arch.latency):
                arch_save.append(arch_prev)
            else:
                arch_save.append(arch)
        return arch_save

    def _expand_width_to_target(self, target):
        arch_save = []
        for op in self.delta_width:
            steplen = 1
            while True:
                arch = self._expand_width(op, steplen)
                if arch.latency >= target:
                    break
                steplen += 1
            arch_prev = self._expand_width(op, steplen - 1)
            if abs(target - arch_prev.latency) < abs(target - arch.latency):
                arch_save.append(arch_prev)
            else:
                arch_save.append(arch)
        return arch_save

    def _expand_resolution_to_target(self, target):
        arch_save = []
        for op in self.delta_resolution:
            steplen = 1
            while True:
                arch = self._expand_resolution(op, steplen)
                if arch.latency >= target:
                    break
                steplen += 1
            arch_prev = self._expand_resolution(op, steplen - 1)
            if abs(target - arch_prev.latency) < abs(target - arch.latency):
                arch_save.append(arch_prev)
            else:
                arch_save.append(arch)
        return arch_save

    def _measure_target_latency(self):
        arch_1step = []
        for op in self.delta_depth:
            arch_1step.append(self._expand_depth(op, 1))
        for op in self.delta_width:
            arch_1step.append(self._expand_width(op, 1))
        for op in self.delta_resolution:
            arch_1step.append(self._expand_resolution(op, 1))
        arch_1step_latency = [a.latency for a in arch_1step]
        return max(arch_1step_latency)

    def expand_all(self):
        print('Measuring target latency...')
        target_latency = self._measure_target_latency()
        print('Expanding from latency {:.4f} to {:.4f}'.format(self.latency, target_latency))
        expanded_arch = []
        expanded_arch.extend(self._expand_depth_to_target(target_latency))
        expanded_arch.extend(self._expand_width_to_target(target_latency))
        expanded_arch.extend(self._expand_resolution_to_target(target_latency))
        # De-duplication
        arch_hash = set()
        unique_archs = []
        for arch in expanded_arch:
            if arch.hash not in arch_hash:
                arch_hash.add(arch.hash)
                unique_archs.append(arch)
        return unique_archs

    def get_slope(self):
        assert self.parent is not None
        assert self.latency > 0
        assert self.miou > 0
        assert self.parent.latency > 0
        assert self.parent.miou > 0
        return (self.miou - self.parent.miou) / (self.latency - self.parent.latency)

    def update_miou(self):
        # Check if miou has already been computed
        if self.miou > -1:
            print('mIoU already computed for this architecture')
            return

        print('Training architecture', self)

        # Create a temporary directory for this experiment
        temp_dir = tempfile.mkdtemp()
        temp_config_path = os.path.join(temp_dir, 'temp_config.py')
        temp_work_dir = temp_dir

        # Path to your base config file
        base_config_path = 'configs/lpsnet/lpsnet_s_400_cityscapes-768x768.py'  # Adjust this path
        
        # Load the base config
        cfg = Config.fromfile(base_config_path)

        # Update the backbone parameters
        cfg.model.backbone.depth = self.depth
        cfg.model.backbone.width = self.width
        cfg.model.backbone.resolution = self.resolution
        
        counter = 0
        for x in self.resolution:
            if x > 0:
                counter += 1
        new_in_channels = self.width[-1] * counter
        cfg.model.decode_head.in_channels = new_in_channels
        
        # Save the updated config
        cfg.dump(temp_config_path)

        print('temp_config_path:', temp_config_path)
        command = [
            'python', 'tools/train.py', temp_config_path,
            '--launcher', 'none',
            '--work-dir', temp_work_dir
        ]
#        command = [
#            'bash', 'tools/dist_train.sh', temp_config_path,
#            '7',
#            '--work-dir', temp_work_dir
#        ]

        # Run the training process
        print('Starting training...')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end='')
        print('Training completed.')
        
        # Find the most recent text log file
        log_files = glob.glob(os.path.join(temp_work_dir, '**', '*.log'), recursive=True)
        if not log_files:
            print('Text log file not found.')
            self.miou = -1
            return
        
        # Assuming the latest log.json is the one we want
        log_file = log_files[-1]
        
        # Parse the log file to get the latest mIoU
        miou = -1
        with open(log_file, 'r') as f:
            for line in f:
                if 'mIoU' in line:
                    # Extract mIoU value using regex
                    import re
                    match = re.search(r'mIoU: ([0-9.]+)', line)
                    if match:
                        miou = float(match.group(1))
        if miou == -1:
            print('mIoU not found in text log file.')
            self.miou = -1
        else:
            self.miou = miou
        
        # Clean up temporary files if desired
        shutil.rmtree(temp_dir)

def initialize_excel(excel_file):
    if os.path.exists(excel_file):
        # Load existing data
        df = pd.read_excel(excel_file)
        print(f'Loaded existing results from {excel_file}')
    else:
        # Create a new DataFrame with the required columns
        columns = ['Step', 'Arch_ID', 'Depth', 'Width', 'Resolution', 'mIoU', 'Latency_ms', 'Slope']
        df = pd.DataFrame(columns=columns)
        print(f'Created new results file at {excel_file}')
    return df

def main():
    # File to save progress
    checkpoint_file = 'progress_checkpoint.pkl'

    # Define results directory and Excel file path
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    excel_file = os.path.join(results_dir, 'architecture_results.xlsx')

    # Initialize or load the Excel file
    df_results = initialize_excel(excel_file)

    # Variables to hold state
    best_arch = []
    step = 0

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_file):
        # Load progress
        with open(checkpoint_file, 'rb') as f:
            best_arch, step = pickle.load(f)
        print(f'Resuming from step {step + 1}')
    else:
        # Starting with a tiny network first
        depth = [1, 3, 3, 10, 10]
        width = [8, 24, 48, 96, 96]
        resolution = [3/4, 1/4, 0]
        arch_init = LPSNetExpansion(depth, width, resolution)
        print('Updated latency: ', arch_init.latency)
        arch_init.update_miou()
        print('Updated miou: ', arch_init.miou)
        best_arch = [arch_init]
        step = 0

        # Log the initial architecture
        arch_id = arch_init.hash  # Using hash as unique ID
        df_results = df_results._append({
            'Step': step + 1,
            'Arch_ID': arch_id,
            'Depth': arch_init.depth,
            'Width': arch_init.width,
            'Resolution': arch_init.resolution,
            'mIoU': arch_init.miou,
            'Latency_ms': arch_init.latency,
            'Slope': float('inf')  # Initial architecture has no slope
        }, ignore_index=True)

        # Save the initial results to Excel
        df_results.to_excel(excel_file, index=False)
        print(f'Logged initial architecture to {excel_file}')

    # Total number of steps
    total_steps = 14

    # Function to handle interrupts
    def signal_handler(sig, frame):
        print('\nProcess interrupted. Saving progress...')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump((best_arch, step), f)
        print('Progress saved. Exiting.')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Expansion steps
    for step in range(step, total_steps):
        print(f'\nExpansion Step {step + 1}')
        base_arch = best_arch[-1]
        arch_expand = base_arch.expand_all()
        print('Measuring miou of expanded architectures...')

        # List to hold data for this step
        step_data = []

        for i, arch in enumerate(arch_expand, 1):
            print(f'Expanded architecture {i}')
            arch.update_miou()
            print('Updated miou: ', arch.miou)
            print('Parent miou: ', arch.parent.miou)
            print('Updated latency: ', arch.latency)
            print('Parent latency: ', arch.parent.latency)
            try:
                slope = arch.get_slope()
                print('Slope: ', slope)
            except AssertionError:
                print('AssertionError: Parent mIoU or latency is invalid.')
                slope = float('-inf')  # Assign a very low slope
            arch.slope = slope  # Optionally, store slope in the arch object

            # Assign unique architecture ID
            arch_id = arch.hash

            # Append data to step_data
            step_data.append({
                'Step': step + 1,
                'Arch_ID': arch_id,
                'Depth': arch.depth,
                'Width': arch.width,
                'Resolution': arch.resolution,
                'mIoU': arch.miou,
                'Latency_ms': arch.latency,
                'Slope': slope
            })

        # Append step_data to the results DataFrame
        df_new = pd.DataFrame(step_data)
        df_results = pd.concat([df_results, df_new], ignore_index=True)
        
        # Save the updated DataFrame to Excel
        df_results.to_excel(excel_file, index=False)
        print(f'Logged step {step + 1} results to {excel_file}')
        
        # Select the best architecture based on slope
        slopes = df_new['Slope'].values
        best_idx = np.argmax(slopes)
        best_arch.append(arch_expand[best_idx])
        print('Current best architecture:')
        print('Depth:', best_arch[-1].depth)
        print('Width:', best_arch[-1].width)
        print('Resolution:', best_arch[-1].resolution)
        
        # Save progress to checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump((best_arch, step + 1), f)
        
        print(f'Progress saved up to step {step + 1}')

    print('\nAll expansion steps completed.')

if __name__ == '__main__':  
    main()
