# GMamba

GMamba is a dual-stream skeleton sequence classification project for
Emotion-Gait Recognition. The model jointly processes:

- **Pose stream**: human joint coordinate sequences.
- **Motion stream**: human movement feature sequences.
- **Affective features**: precomputed affective features used as auxiliary
  supervision and model conditioning.

The main model combines Adaptive Graph Convolution, multi-scale temporal
modeling, bidirectional Mamba blocks, part-aware gating, cross-stream fusion,
affective feature conditioning, and optional layer-wise contrastive learning.

## Project Structure

```text
GMamba/
|-- config/
|   `-- train_gmamba.yaml          # Training configuration
|-- datasets/
|   `-- emotion_gait/              # Emotion-Gait data files
|-- feeders/
|   |-- feeder_emotion_gait.py     # Dataset loader
|   `-- tools.py                   # Padding and augmentation utilities
|-- model/
|   |-- gmamba.py                  # GMamba model
|   `-- loss_fusion_modules.py     # Auxiliary loss modules
|-- main.py                        # Training, testing, and export entry point
|-- requirements.yml               # Conda environment specification
`-- README.md
```

## Requirements

The recommended environment is:

- Python 3.10
- PyTorch 2.1.1
- CUDA 11.8
- NVIDIA GPU with 8 GB or more of memory

The provided `requirements.yml` was exported from a Linux/CUDA environment.
On Linux or WSL, create the environment with:

```bash
conda env create -f requirements.yml
conda activate gaitmamba
```

For a native Windows environment, create a clean Python 3.10 environment:

```powershell
conda create -n gaitmamba python=3.10 -y
conda activate gaitmamba
```

For CUDA 11.8, install PyTorch and the main Python dependencies with:

```powershell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 pyyaml matplotlib seaborn tqdm einops
```

The model uses `mamba-ssm` when it is available:

```bash
pip install causal-conv1d==1.1.3.post1 mamba-ssm==1.1.1
```

If `mamba-ssm` is unavailable or cannot be built on the current platform,
GMamba automatically uses the built-in `MambaLite` implementation. You can
also disable the external implementation explicitly:

```yaml
use_mamba_ssm: false
```

## Dataset

The project expects the Emotion-Gait data under
`datasets/emotion_gait/`. The directory is kept in the project structure, but
the data files may need to be downloaded or copied separately.

| File | Shape | Description |
| --- | --- | --- |
| `train_joint.npy` | `(1959, 3, 48, 16, 1)` | Training pose sequences |
| `train_movement.npy` | `(1959, 8, 48, 16, 1)` | Training motion sequences |
| `train_affective.npy` | `(1959, 1488)` | Training affective features |
| `train_label.pkl` | 1,959 labels | Training labels |
| `test_joint.npy` | `(218, 3, 48, 16, 1)` | Test pose sequences |
| `test_movement.npy` | `(218, 8, 48, 16, 1)` | Test motion sequences |
| `test_affective.npy` | `(218, 1488)` | Test affective features |
| `test_label.pkl` | 218 labels | Test labels |

The dimensions use the following convention:

```text
N: number of samples
C: number of channels
T: number of frames, 48 by default
V: number of joints, 16 by default
M: number of people, 1 by default
```

The default task has four classes:

```text
0: Happy
1: Sad
2: Angry
3: Neutral
```

The label file is a pickle file containing a pair of sample names and labels.
The NumPy files are loaded with memory mapping by default.

## Quick Start

Run all commands from the project root:

```text
GMamba/
```

### Important Configuration Note

The current `config/train_gmamba.yaml` contains two module paths from an older
version of the project:

```yaml
feeder: feeders.feeder_same_combine.Feeder
model: model.gmamba_test.GMamba
```

The modules currently available in this repository are:

```text
feeders.feeder_emotion_gait.Feeder
model.gmamba.GMamba
```

Therefore, pass the current paths explicitly on the command line, or update
the two fields in the YAML file.

### Training

Run a complete training job with the provided configuration:

```bash
python main.py --config config/train_gmamba.yaml --feeder feeders.feeder_emotion_gait.Feeder --model model.gmamba.GMamba
```

PowerShell example:

```powershell
python main.py --config config/train_gmamba.yaml --feeder feeders.feeder_emotion_gait.Feeder --model model.gmamba.GMamba --device 0 --num-worker 0
```

The default configuration trains for 120 epochs and writes results to:

```text
work_dir/lr_test/
runs/lr_test/
```

### Smoke Test

Use the following command to verify data loading and one training/evaluation
step without running a full experiment:

```bash
python main.py --config config/train_gmamba.yaml --feeder feeders.feeder_emotion_gait.Feeder --model model.gmamba.GMamba --num-worker 0 --num-epoch 1 --max-train-batches 1 --max-eval-batches 1 --export-confusion false --export-analysis-data false --export-visuals false
```

### Testing a Saved Model

First enable model saving during training:

```bash
python main.py --config config/train_gmamba.yaml --feeder feeders.feeder_emotion_gait.Feeder --model model.gmamba.GMamba --save_model true
```

The best checkpoint is saved as:

```text
./runs/lr_test/best_PG_DNDT_CSTA.pt
```

Evaluate a saved checkpoint with:

```bash
python main.py --phase test --config config/train_gmamba.yaml --feeder feeders.feeder_emotion_gait.Feeder --model model.gmamba.GMamba --weights ./runs/lr_test/best_PG_DNDT_CSTA.pt --model-saved-name ./runs/lr_test_test --work-dir ./work_dir/lr_test_test
```

During training, the program creates the configured working and run
directories. Typical files include:

```text
work_dir/
|-- config.yaml       # Resolved parameters for the current run
|-- log.txt           # Training and evaluation log
|-- main.py           # Backup of the entry script
`-- gmamba.py         # Backup of the model file

runs/
|-- train/            # TensorBoard training events, when tensorboardX is installed
|-- val/              # TensorBoard validation events
|-- best_PG_DNDT_CSTA.pt
|-- confusion/
|   |-- confusion_best.npz
|   |-- confusion_best_raw.csv
|   |-- confusion_best_normalized.csv
|   |-- confusion_best_classification_results.csv
|   `-- confusion_best_summary.txt
|-- visuals/          # Created when export_visuals is enabled
`-- analysis_data/    # Created when export_analysis_data is enabled
```

The confusion-matrix export contains accuracy, precision, recall, F1,
G-mean, Kappa, sample indices, true labels, predicted labels, logits, and
class probabilities.

To enable TensorBoard logging:

```bash
pip install tensorboardX
tensorboard --logdir ./runs/lr_test
```

## Development Checks

Check Python syntax:

```bash
python -m py_compile main.py model/gmamba.py model/loss_fusion_modules.py feeders/feeder_emotion_gait.py feeders/tools.py
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Notes

1. Run commands from the project root so that the relative dataset paths in
   the configuration resolve correctly.
2. With `num_point=16`, the model uses the Emotion-Gait-specific joint
   topology. Other joint counts use a fallback chain topology.
3. `emotion_feature_dim` must match the flattened dimension of the affective
   feature files. The provided data uses 1488.
4. The default `main.py` parser points to
   `config/train_gmamba_origin.yaml`, which is not included in this
   repository. Always pass `--config config/train_gmamba.yaml`.
5. The visualization and stage-analysis exports are optional and disabled in
   the provided configuration. Enable them only after confirming that the
   corresponding intermediate outputs are available in the selected model
   implementation.

## License and Citation

This repository currently does not include an explicit license or BibTeX
entry. Before publishing or using the project in an academic work, add the
appropriate license, dataset attribution, and paper citation information.
