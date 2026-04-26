# Training on Brown Oscar (CCV)

Step-by-step setup for training the pitch detector on Brown's Oscar SLURM cluster.

## 1. Find out which partition you can use

```bash
ssh <YOUR_USERNAME>@ssh.ccv.brown.edu

# Check what partitions you can submit to
sacctmgr show assoc user=$USER format=Account,Partition,QOS

# Common partitions:
#   gpu        — standard GPU partition (V100/3090/A40 mix), QoS=norm-gpu
#   gpu-debug  — short jobs only, fast turnaround for debugging
#   gpu-he     — high-end (A100s), if you have access
```

The default SLURM scripts target `gpu` with `--qos=norm-gpu`. If you have `gpu-he`,
edit `scripts/oscar_train.sbatch` to use it for ~2× faster training.

## 2. Push the project + dataset to Oscar

From your local machine, after creating the destination dir on Oscar:

```bash
ssh <USER>@ssh.ccv.brown.edu "mkdir -p ~/scratch/'JadeDragon Transcriber'"

# Note: backslash-escape the space on the remote side, otherwise rsync silently
# splits the path and lands files at ~/scratch/JadeDragon/ (without "Transcriber").
rsync -avz --progress \
    --exclude='outputs/' \
    --exclude='runs/' \
    --exclude='checkpoints/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='Gongche notation dataset/' \
    "/path/to/JadeDragon Transcriber/" \
    "<USER>@ssh.ccv.brown.edu:scratch/JadeDragon\ Transcriber/"
```

The dataset (`LGRC2024 dataset/`) ships with this rsync (~1 GB). After the
first push, code edits sync incrementally with the same command.

## 3. Set up the conda environment on Oscar

**Oscar doesn't ship a generic `anaconda` or `miniconda` module.** Install
miniforge into your home directory once (no sudo needed):

```bash
ssh <USER>@ssh.ccv.brown.edu

curl -L -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc

which conda           # should print $HOME/miniforge3/bin/conda
conda --version
```

Then build the project env:

```bash
cd ~/scratch/'JadeDragon Transcriber'
conda env create -f environment.yml      # 5–10 min
conda activate jadedragon

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# CUDA: False on the login node is fine — the smoke job verifies it on a real GPU node.
```

The SLURM scripts (`oscar_smoke.sbatch`, `oscar_train.sbatch`) source
`$HOME/miniforge3/etc/profile.d/conda.sh` directly, so once miniforge is
installed they work without any module-loading fuss.

## 4. Submit the smoke test (5 epochs, ~10 min)

Always smoke-test before queueing the full job:

```bash
mkdir -p logs
sbatch scripts/oscar_smoke.sbatch
squeue -u $USER

# Once it has a job ID, watch the log:
tail -f logs/jadedragon-smoke-<JOBID>.out
```

The smoke job:
1. Reports CUDA + GPU info
2. Runs dataset prep (idempotent — creates `lgrc2024_yolo/` with 0-indexed labels)
3. Trains YOLOv8m for 5 epochs (~3 min on A40)
4. Reloads the trained checkpoint to confirm it's valid

## 5. Submit the full training (150 epochs, ~1 h on A40)

```bash
sbatch scripts/oscar_train.sbatch

# Override defaults via env vars if you want:
#   EPOCHS=50 BATCH=32 RUN_NAME=ablation_short sbatch scripts/oscar_train.sbatch
```

Best weights end up at `checkpoints/lgrc2024_yolov8m_best.pt` when the job finishes.

## 6. Pull weights back to your Mac

```bash
mkdir -p ~/Downloads/'JadeDragon Transcriber'/checkpoints

scp '<USER>@ssh.ccv.brown.edu:/oscar/scratch/<USER>/JadeDragon\ Transcriber/checkpoints/lgrc2024_yolov8m_best.pt' \
    ~/Downloads/'JadeDragon Transcriber'/checkpoints/

# Optionally also pull the training plots
scp '<USER>@ssh.ccv.brown.edu:/oscar/scratch/<USER>/JadeDragon\ Transcriber/runs/detect/lgrc2024_yolov8m/results.png' \
    ~/Downloads/'JadeDragon Transcriber'/outputs/
```

Inference is small enough to run on the Mac CPU/MPS — no GPU needed locally.

## Notes / gotchas (lessons learned)

- **Spaces in path names**: Always backslash-escape on the remote side of rsync.
- **First Ultralytics run**: Auto-downloads `yolov8m.pt` (~50 MB). Compute nodes
  on Oscar have internet, so this Just Works on first batch run.
- **Resume from checkpoint**: If a SLURM job hits the time limit before finishing,
  resume with `python scripts/train_pitch_detector.py --model runs/detect/lgrc2024_yolov8m/weights/last.pt`.
- **Disable training augmentations**: We turn off rotate/flip in
  `train_pitch_detector.py` — they break gongche reading order. This gave us a
  ~7 mAP50 improvement over the LGRC2024 paper's vanilla YOLOv8m baseline.
- **SLURM scripts use `set -eo pipefail` (no `-u`)**: conda activate scripts
  reference unset env vars (`MKL_INTERFACE_LAYER` etc.); strict `-u` crashes the job.
- **Early stopping**: Ultralytics' default `patience=100` epochs without
  improvement. Our peak val mAP comes around epoch 25–30, so the last ~80 epochs
  are wasted. Lower `patience` for faster turnaround.
