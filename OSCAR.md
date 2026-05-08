# Training on Brown Oscar (CCV)

How to train the YOLO detectors on Brown's Oscar SLURM cluster.

This project trains two detectors on Oscar:

- **Pitch detector** (28-class) on the LGRC2024 dataset (~1496 train images) — `scripts/oscar_train.sbatch`
- **Lyric detector** (single-class) on the JadeDragon lyric dataset (~40 train images, hand-labeled in VIA) — `scripts/oscar_train_lyric.sbatch`

Both go through the same SSH/SLURM workflow. The pitch detector takes ~1 hour on an A40; the lyric detector takes ~10-15 min on an L40S because the dataset is much smaller.

---

## Setup (one-time)

### 1. Get an account and check your partitions

```bash
ssh ssh.ccv.brown.edu

sacctmgr show assoc user=$USER format=Account,Partition,QOS
```

Common partitions:
- `gpu` — standard mix (V100 / 3090 / A40 / L40S), QoS `norm-gpu`
- `gpu-debug` — short jobs, fast turnaround
- `gpu-he` — A100s (if you have access)

The default sbatch scripts target `gpu`. If you have `gpu-he`, edit them for ~2× faster training.

### 2. Set up an SSH alias with ControlMaster

ControlMaster lets you reuse a single authenticated SSH session for ~8 hours, so you don't have to re-enter the OTP for every command. Add this to `~/.ssh/config` on your Mac:

```
Host oscar
    HostName ssh.ccv.brown.edu
    User <YOUR_USERNAME>
    ControlMaster auto
    ControlPath ~/.ssh/oscar-%r@%h:%p
    ControlPersist 8h
```

After that, every `ssh oscar`, `scp ... oscar:...`, and `rsync ... oscar:...` reuses the same connection — auth once, run anything for the next 8 hours. To force a fresh connection: `ssh -O exit oscar`.

### 3. Pick where to store the project on Oscar

Two reasonable locations:

| Location | Use case | Caveat |
|---|---|---|
| `/oscar/data/class/<COURSE>/students/<USER>/JadeDragon/` | Class-affiliated project | Only available while course is active |
| `~/scratch/JadeDragon/` | Personal | Files older than 30 days are auto-deleted |

For this project we use `/oscar/data/class/csci1430/students/zyang188/JadeDragon/`. Adjust to match your account.

### 4. Push the project to Oscar

From your Mac:

```bash
# Create destination on Oscar
ssh oscar "mkdir -p /oscar/data/class/csci1430/students/$USER/JadeDragon"

# Push code + data, excluding the giant excluded dirs
rsync -avh --progress \
    --exclude='outputs/' \
    --exclude='runs/' \
    --exclude='checkpoints/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='Gongche notation dataset/' \
    "/Users/kellyyang/Downloads/JadeDragon Transcriber/" \
    "oscar:/oscar/data/class/csci1430/students/$USER/JadeDragon/"
```

The `LGRC2024 dataset/` directory ships with this rsync (~1 GB). After the first push, code edits sync incrementally with the same command.

For lyric-detector retraining you can sync just the lyric dataset:

```bash
rsync -avh --delete lyric_dataset/ \
    "oscar:/oscar/data/class/csci1430/students/$USER/JadeDragon/lyric_dataset/"
```

The `--delete` flag mirrors deletions from your Mac to Oscar, useful when you've removed images from the dataset.

### 5. Set up the Python environment on Oscar

This project uses a `.venv` virtualenv on Oscar (NOT conda). One-time setup:

```bash
ssh oscar
cd /oscar/data/class/csci1430/students/$USER/JadeDragon

# Use the system Python 3.11+
module load python/3.13      # or whatever's available
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Sanity check
python -c "import torch; print('CUDA build:', torch.cuda.is_available())"
# CUDA: False on the login node is fine — the SLURM job runs on a GPU node.
```

The sbatch scripts source `.venv/bin/activate` directly, so once it's set up the SLURM jobs Just Work.

---

## Submitting training jobs

### Pitch detector (full training, ~1 h on A40)

```bash
ssh oscar
cd /oscar/data/class/csci1430/students/$USER/JadeDragon
mkdir -p logs
sbatch scripts/oscar_train.sbatch
squeue -u $USER

# Watch progress
tail -f logs/jadedragon-pitch-<JOBID>.out
```

Best weights land at `checkpoints/lgrc2024_yolov8m_best.pt` on Oscar.

### Pitch detector (smoke test, ~3 min)

Always smoke-test before queueing the full job:

```bash
sbatch scripts/oscar_smoke.sbatch
```

The smoke job runs 5 epochs, then reloads the checkpoint to confirm it's valid.

### Lyric detector (~10-15 min on L40S)

After labeling new pages on your Mac and pushing the updated `lyric_dataset/`:

```bash
# (after rsync of lyric_dataset/ — see step 4)
ssh oscar "cd /oscar/data/class/csci1430/students/$USER/JadeDragon && sbatch scripts/oscar_train_lyric.sbatch"
ssh oscar "squeue -u $USER"
```

Best weights land at `checkpoints/lyric_yolov8m_best.pt` on Oscar. The sbatch script auto-copies `runs/detect/lyric_yolov8m/weights/best.pt` to `checkpoints/` after training.

If a previous training run is still in `runs/detect/lyric_yolov8m/`, ultralytics auto-increments to `lyric_yolov8m-2/`, `lyric_yolov8m-3/`, etc. — the `best.pt` you want is in the most recently modified subdirectory.

---

## Pulling weights back to your Mac

```bash
cd "/Users/kellyyang/Downloads/JadeDragon Transcriber"

# Pitch
scp 'oscar:/oscar/data/class/csci1430/students/zyang188/JadeDragon/checkpoints/lgrc2024_yolov8m_best.pt' \
    checkpoints/lgrc2024_yolov8m_best.pt

# Lyric (from the script's auto-copied location)
scp 'oscar:/oscar/data/class/csci1430/students/zyang188/JadeDragon/checkpoints/lyric_yolov8m_best.pt' \
    checkpoints/lyric_yolov8m_best.pt

# Optional: training plots for reports/posters
scp 'oscar:/oscar/data/class/csci1430/students/zyang188/JadeDragon/runs/detect/lyric_yolov8m-3/results.png' \
    outputs/lyric_results.png
scp 'oscar:/oscar/data/class/csci1430/students/zyang188/JadeDragon/runs/detect/lyric_yolov8m-3/BoxF1_curve.png' \
    outputs/lyric_F1_curve.png
```

Inference runs fine on the Mac CPU/MPS — no GPU needed locally for the main pipeline.

---

## Diagnosing failed jobs

```bash
# What's the job's exit status?
ssh oscar "sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem -P"

# Read the actual error
ssh oscar "cat logs/jadedragon-lyric-<JOBID>.err"

# How many lines did the .out file get?
ssh oscar "wc -l logs/jadedragon-lyric-<JOBID>.out"
```

Common failure modes:

- **`State=FAILED, ExitCode=1:0, Elapsed=00:00:23`** + cryptic stack trace — usually a path resolution issue in `data.yaml` or missing dataset file. Check the `.err` file.
- **`State=OUT_OF_MEMORY`** — ask for more memory: edit `#SBATCH --mem=` in the sbatch file.
- **`State=TIMEOUT`** — bump `#SBATCH --time=` (default in sbatch scripts is conservative).
- **`FileNotFoundError ... images not found`** — `data.yaml` has a relative `path:` that resolves wrong. See the data.yaml gotcha below.

### The `data.yaml` gotcha

Ultralytics resolves `path:` in `data.yaml` relative to the current working directory of the training process, NOT relative to the YAML file itself. If `data.yaml` says `path: ./` and SLURM runs from the project root, ultralytics looks for `./images/val/` at the project root — which doesn't exist; the images live at `lyric_dataset/images/val/`.

Fix: use an absolute path on Oscar:

```yaml
# /oscar/.../JadeDragon/lyric_dataset/data.yaml
path: /oscar/data/class/csci1430/students/zyang188/JadeDragon/lyric_dataset
train: images/train
val: images/val
nc: 1
names:
  0: lyric
```

You can keep the Mac copy with relative paths if you always train from the lyric_dataset dir — but easier to use absolute paths in both places (with Mac-specific path on Mac, Oscar-specific path on Oscar) and not let them sync via rsync. Or just make Oscar's data.yaml absolute and let the Mac one fend for itself.

---

## Notes / gotchas (lessons learned)

- **Spaces in path names**: avoid them in remote paths. The current convention `/oscar/.../JadeDragon/` (no space) was deliberately chosen so rsync and scp don't need backslash-escapes.
- **First Ultralytics run**: Auto-downloads `yolov8m.pt` (~50 MB) from PyTorch Hub. Compute nodes have internet, so this Just Works on first batch run.
- **Resume from checkpoint**: if a SLURM job hits the time limit before finishing, edit the train script to add `--resume runs/detect/<NAME>/weights/last.pt` and resubmit.
- **Disable training augmentations for gongche detection**: rotate/flip break gongche reading order. The pitch training script disables them; this gave us ~7 mAP50 over the LGRC2024 paper's vanilla YOLOv8m baseline.
- **Early stopping**: Ultralytics' default `patience=100` is too long for our datasets. The pitch detector peaks around epoch 25–30; the lyric detector peaks around 18. We set `patience=20` in the sbatch scripts.
- **Multiple training runs auto-increment**: if `runs/detect/lyric_yolov8m/` already exists, ultralytics writes to `lyric_yolov8m-2/`, then `lyric_yolov8m-3/`, etc. When pulling weights, list the dir first to make sure you grab the latest:
  ```bash
  ssh oscar "ls -t runs/detect/ | head -5"
  ```
- **Empty weights dir after a "successful" SLURM run** = the training crashed silently. Read the `.err` file. Common cause: data.yaml path resolution.
- **L40S vs A40**: lyric training jobs sometimes land on either depending on what's available. L40S (46 GB VRAM) is faster; A40 (48 GB) is fine. RTX 3090 (24 GB) also works but is slower. The sbatch script doesn't pin a specific card type — SLURM picks whatever is free.
