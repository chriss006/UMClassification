#!/bin/bash
set -euo pipefail
cd /home/halee/projects/UMClassification
source /home/halee/venv310/bin/activate
python finetune.py --config swint.swin_tiny_block_cbam_all --kfold 3 --fold_index 2
