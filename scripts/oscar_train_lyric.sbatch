#!/bin/bash
#SBATCH --job-name=jadedragon-lyric
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL

# JadeDragon Transcriber — train the LYRIC detector (single class) on Oscar.
#
# Submit:   sbatch scripts/oscar_train_lyric.sbatch
# Watch:    myq          (or: squeue -u $USER)
# Logs:     logs/jadedragon-lyric-<jobid>.out
#
# Prerequisites:
#   1. lyric_dataset/ already populated (via VIA + scripts/via_to_yolo.py).
#   2. uv installed at $HOME/.local/bin/uv (curl -LsSf https://astral.sh/uv/install.sh | sh).
#   3. Project files synced to $PROJECT_DIR (see rsync command in CLAUDE.md).

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/oscar/data/class/csci1430/students/zyang188/JadeDragon}"
DATA_YAML="${DATA_YAML:-$PROJECT_DIR/lyric_dataset/data.yaml}"
RUN_NAME="${RUN_NAME:-lyric_yolov8m}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-20}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
# Set MODEL=checkpoints/lgrc2024_yolov8m_best.pt to warm-start from the pitch
# detector's backbone instead of ImageNet weights.
MODEL="${MODEL:-yolov8m.pt}"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

if [ ! -f "$DATA_YAML" ]; then
  echo "ERROR: lyric data.yaml not found at $DATA_YAML"
  echo "Build the dataset first with scripts/via_to_yolo.py via_all.json"
  exit 1
fi

# Activate the project's uv-managed virtualenv (created once via:
#   uv venv && uv pip install ultralytics music21 pillow numpy opencv-python pyyaml)
if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
  echo "ERROR: virtualenv not found at $PROJECT_DIR/.venv"
  echo "Set it up once with:"
  echo "  cd $PROJECT_DIR"
  echo "  uv venv"
  echo "  uv pip install ultralytics music21 pillow numpy opencv-python pyyaml"
  exit 1
fi
source "$PROJECT_DIR/.venv/bin/activate"

if ! python -c "import ultralytics" 2>/dev/null; then
  echo "ERROR: ultralytics missing from venv. Run inside venv:"
  echo "  uv pip install ultralytics music21 pillow numpy opencv-python pyyaml"
  exit 1
fi

echo "Starting lyric-detector training: $RUN_NAME"
echo "  Project: $PROJECT_DIR"
echo "  Data:    $DATA_YAML"
echo "  Base:    $MODEL"
echo "  Epochs:  $EPOCHS, patience: $PATIENCE, batch: $BATCH, imgsz: $IMGSZ"
echo "  Python:  $(which python)"
nvidia-smi || true

python scripts/train_lyric_detector.py \
    --data "$DATA_YAML" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    --name "$RUN_NAME" \
    --device 0

mkdir -p "$PROJECT_DIR/checkpoints"
cp "runs/detect/$RUN_NAME/weights/best.pt" "$PROJECT_DIR/checkpoints/${RUN_NAME}_best.pt"
echo "Best weights copied to checkpoints/${RUN_NAME}_best.pt"
echo "Pull to Mac with:"
echo "  scp 'oscar:$PROJECT_DIR/checkpoints/${RUN_NAME}_best.pt' \\"
echo "      '/Users/kellyyang/Downloads/JadeDragon Transcriber/checkpoints/'"
