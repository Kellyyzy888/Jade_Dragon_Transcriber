#!/bin/bash
#SBATCH --job-name=jadedragon-yolov8m
#SBATCH --partition=gpu                 # zyang188 has access to: gpu, gpu-debug
#SBATCH --qos=norm-gpu                  # standard QoS for the gpu partition
#SBATCH --gres=gpu:1                    # 1 GPU (Oscar's gpu partition is V100/3090/A40 mix)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00                 # 8 hours — plenty for 150 epochs of YOLOv8m on ~2k images on V100/3090
#SBATCH --output=logs/%x-%j.out         # %x=job-name, %j=jobid
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL            # email on completion/failure (optional)

# JadeDragon Transcriber: train YOLOv8m on LGRC2024.
#
# Submit with:    sbatch scripts/oscar_train.sbatch
# Check:          squeue -u $USER
# Cancel:         scancel <jobid>
# Logs:           logs/jadedragon-yolov8m-<jobid>.out / .err

set -eo pipefail   # NOTE: -u dropped because conda activate scripts reference unset vars (e.g. MKL_INTERFACE_LAYER)

# ---- Paths --------------------------------------------------------------------
# Adjust these to match your Oscar layout. The defaults assume you rsync'd the
# project into your scratch space at ~/scratch/JadeDragon\ Transcriber/.

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/JadeDragon Transcriber}"
DATA_SRC="${DATA_SRC:-$PROJECT_DIR/LGRC2024 dataset/datasets}"
DATA_REMAP="${DATA_REMAP:-$PROJECT_DIR/lgrc2024_yolo}"   # 0-indexed copy
RUN_NAME="${RUN_NAME:-lgrc2024_yolov8m}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# ---- Environment --------------------------------------------------------------
# We use a user-installed miniforge in $HOME/miniforge3 (Oscar doesn't ship a
# generic `anaconda` or `miniconda` module). One-time install:
#   bash <(curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) -b -p $HOME/miniforge3
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate jadedragon

# ---- One-time data prep (idempotent) -----------------------------------------
# Rewrite labels to 0-indexed and symlink images into a parallel tree.
# Skipped if `$DATA_REMAP/labels/train` already exists.
if [ ! -d "$DATA_REMAP/labels/train" ]; then
  echo "Remapping LGRC2024 labels to 0-indexed at $DATA_REMAP ..."
  python scripts/prepare_dataset.py --src "$DATA_SRC" --dst "$DATA_REMAP" --remap --symlink
fi

# ---- Build the data.yaml the trainer will use --------------------------------
DATA_YAML="$PROJECT_DIR/configs/lgrc2024_oscar.yaml"
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

# ---- Train --------------------------------------------------------------------
echo "Starting training: $RUN_NAME"
echo "  Epochs: $EPOCHS, batch: $BATCH, imgsz: $IMGSZ"
nvidia-smi || true

python scripts/train_pitch_detector.py \
    --data "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    --name "$RUN_NAME" \
    --device 0

# ---- Stash best checkpoint somewhere stable ----------------------------------
mkdir -p "$PROJECT_DIR/checkpoints"
cp "runs/detect/$RUN_NAME/weights/best.pt" "$PROJECT_DIR/checkpoints/${RUN_NAME}_best.pt"
echo "Best weights copied to checkpoints/${RUN_NAME}_best.pt"

echo "Job complete. Use:"
echo "  scp 'oscar:$PROJECT_DIR/checkpoints/${RUN_NAME}_best.pt' ~/Downloads/JadeDragon\\ Transcriber/checkpoints/"
echo "to pull the trained model back to your Mac for the demo notebook."
