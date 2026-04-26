#!/bin/bash
#SBATCH --job-name=jadedragon-smoke
#SBATCH --partition=gpu-debug           # short queue, fast turnaround
#SBATCH --qos=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00                 # 30 minutes — well under gpu-debug limit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# JadeDragon smoke test: verify the env, GPU, dataset, and YOLO training loop
# all work end-to-end before kicking off the 150-epoch run.
#
# What it does:
#   1. Reports CUDA + GPU info
#   2. Runs the dataset preparation step (idempotent)
#   3. Trains YOLOv8m for 5 epochs (a few minutes) on the LGRC2024 train set
#   4. Validates the trained checkpoint loads back
#
# Submit with:    sbatch scripts/oscar_smoke.sbatch
# Total runtime should be 5–10 minutes.

set -eo pipefail   # NOTE: -u dropped because conda activate scripts reference unset vars (e.g. MKL_INTERFACE_LAYER)

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/JadeDragon Transcriber}"
DATA_SRC="${DATA_SRC:-$PROJECT_DIR/LGRC2024 dataset/datasets}"
DATA_REMAP="${DATA_REMAP:-$PROJECT_DIR/lgrc2024_yolo}"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# ---- Env ----------------------------------------------------------------------
# We use a user-installed miniforge (Oscar doesn't ship a generic `anaconda` module).
# Install once with: bash <(curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) -b -p $HOME/miniforge3
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate jadedragon

# ---- Diagnostics --------------------------------------------------------------
echo "==== node ===="
hostname
echo "==== nvidia-smi ===="
nvidia-smi || echo "nvidia-smi not available"
echo "==== python / pytorch ===="
python -c "
import torch, sys
print('python:', sys.version)
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Capability:', torch.cuda.get_device_capability(0))
"
echo "==== ultralytics ===="
python -c "from ultralytics import YOLO; print('ultralytics OK')"
echo "==== music21 ===="
python -c "from music21 import converter; print('music21 OK')"

# ---- Dataset prep -------------------------------------------------------------
if [ ! -d "$DATA_REMAP/labels/train" ]; then
  echo "==== prepare_dataset (one-time) ===="
  python scripts/prepare_dataset.py --src "$DATA_SRC" --dst "$DATA_REMAP" --remap --symlink
fi

# Build a smoke-test data.yaml on the fly
DATA_YAML="$PROJECT_DIR/configs/lgrc2024_oscar.yaml"
if [ ! -f "$DATA_YAML" ]; then
  cat > "$DATA_YAML" <<EOF
path: $DATA_REMAP
train: images/train
val: images/val
nc: 28
names:
  0: keySignature-bB
  1: keySignature-C
  2: keySignature-D
  3: keySignature-bE
  4: keySignature-F
  5: keySignature-G
  6: keySignature-A
  7: note-C3
  8: note-D3
  9: note-E3
  10: note-F3
  11: note-G3
  12: note-A3
  13: note-B3
  14: note-C4
  15: note-D4
  16: note-E4
  17: note-F4
  18: note-G4
  19: note-A4
  20: note-B4
  21: note-C5
  22: note-D5
  23: note-E5
  24: note-F5
  25: note-G5
  26: note-A5
  27: note-B5
EOF
fi

# ---- 5-epoch training run -----------------------------------------------------
echo "==== smoke training: 5 epochs ===="
python scripts/train_pitch_detector.py \
    --data "$DATA_YAML" \
    --epochs 5 \
    --batch 8 \
    --imgsz 640 \
    --name smoke_test \
    --device 0

# ---- Verify checkpoint loads --------------------------------------------------
echo "==== reload checkpoint ===="
python -c "
from ultralytics import YOLO
m = YOLO('runs/detect/smoke_test/weights/best.pt')
print('Loaded:', m.model.__class__.__name__)
print('Classes:', m.names)
"

echo "==== smoke test complete ===="
