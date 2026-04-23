#!/usr/bin/env python3
"""Custom CLAM training script for TMB binary classification."""

import os
import sys
import subprocess

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to CLAM directory
clam_dir = os.path.join(script_dir, 'CLAM')
os.chdir(clam_dir)

# Run main.py with custom TMB task
cmd = [
    'python', 'main.py',
    '--drop_out', '0.25',
    '--early_stopping',
    '--lr', '2e-4',
    '--k', '1',
    '--exp_code', 'task_3_tmb_binary_CLAM_100',
    '--weighted_sample',
    '--bag_loss', 'ce',
    '--inst_loss', 'svm',
    '--task', 'task_3_tmb_binary',
    '--model_type', 'clam_sb',
    '--log_data',
    '--data_root_dir', '..',
    '--embed_dim', '1280',
    '--split_dir', 'task_3_tmb_binary_100'
]

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'

result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)
