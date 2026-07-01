# DDM2 for Brain CT (DDM2_brainct)

A self-supervised diffusion denoiser adapted from **DDM2** (*Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models*, ICLR 2023) for **brain CT** low-dose / simulated-noise denoising. This repo is used as a **baseline** method.

## Overview

DDM2 denoises without clean ground truth by using a 3-stage self-supervised pipeline:

1. **Stage 1 — Noise model** (`run_stage1.sh`): trains a network to model the noise/initial estimate on the noisy CT volumes.
2. **Stage 2 — State matching** (`run_stage2.sh`): matches each noisy slice to a diffusion timestep, producing `stage2_matched.txt` used to condition the diffusion training.
3. **Stage 3 — Diffusion denoiser** (`run_stage3.sh`): trains the conditional diffusion U-Net that performs the actual denoising.

A Noise2Noise (N2N) result is used as a teacher/conditioning image (`teacher_n2n_root`), and histogram equalization (HE) is applied in the data pipeline with pre-computed bins (inverse HE is applied at inference to return to HU space).

## Key features

- Self-supervised: no clean CT ground truth required for training.
- Conditional diffusion U-Net (1-channel, 512x512), `n_timestep=1000`, `rev_warmup70` beta schedule.
- Histogram-equalization preprocessing with invertible bins mapping (`bins.npy` / `bins_mapped.npy`).
- N2N teacher used as the conditioning image.
- Full-volume `nii.gz` inference with automatic HE inversion back to HU.

## Repo structure

- `config/ct_denoise.json` — single config for all 3 stages (paths, HU window, HE, slice range, teacher N2N, model, beta schedule).
- `train_noise_model.py` — Stage 1 training (noise model).
- `match_state.py` — Stage 2 state matching; writes `stage2_file` (`experiments/ct_denoise_teacher/stage2_matched.txt`).
- `train_diff_model.py` — Stage 3 training (conditional diffusion denoiser).
- `run_stage1.sh` / `run_stage2.sh` / `run_stage3.sh` — thin wrappers around the three scripts above.
- `inference_ddm2.py` — full-volume inference; outputs `nii.gz` with optional inverse HE.
- `eval_ddm2_local.py` — local evaluation (MAE, SSIM, LPIPS) of DDM2 vs Noisy and vs N2N; writes results xlsx.
- `nii2npy.py` — optional helper converting `nii.gz` to `npy` to speed up training/inference.
- `denoise.py` / `denoise.sh` — standalone denoising entry point.
- `sample.py` / `sample.sh` — sampling utility (references upstream `hardi_150.json` config).
- `test.py` — test/eval entry point.
- `metrics.py`, `quantitative_metrics.ipynb` — metric utilities / analysis notebook.
- `check_data.py` — quick data sanity check.
- `data/` — dataset code: `ct_dataset.py` (CT dataset with HE + slice-offset detection), `prepare_data.py`, `util.py`.
- `model/` — models: `model.py`, `model_stage1.py`, `noise_model.py`, `base_model.py`, `networks.py`, and `mri_modules/` (`diffusion.py`, `noise_model.py`, `unet.py`, `simple_unet.py`, `utils.py`).
- `core/` — `logger.py`, `metrics.py`.
- `experiments/` — training/inference outputs (checkpoints, logs, `inference/`).
- `environment.yml` — conda environment.

## Setup

```bash
conda env create -f environment.yml
```

Edit `config/ct_denoise.json` and set the local paths: `dataroot` (xlsx list), `data_root` (noisy volumes), `bins_file` / `bins_mapped_file` (HE), and `teacher_n2n_root` (N2N predictions). Optionally run `nii2npy.py` on the noisy and N2N folders to cache `npy` and speed up I/O.

## How to run

All three stages share the same config and use `-p train -c config/ct_denoise.json`.

```bash
# Stage 1: train noise model
./run_stage1.sh
# = python3 train_noise_model.py -p train -c config/ct_denoise.json

# Stage 2: state matching -> writes experiments/ct_denoise_teacher/stage2_matched.txt
./run_stage2.sh
# = mkdir -p experiments/ct_denoise_teacher && python3 match_state.py -p train -c config/ct_denoise.json

# Stage 3: train diffusion denoiser
./run_stage3.sh
# = python3 train_diff_model.py -p train -c config/ct_denoise.json
```

### Inference

```bash
python inference_ddm2.py -c config/ct_denoise.json --patient_idx 0
# optional: --checkpoint <path> --output_dir <dir> --gpu 0 --no_inverse_he
```

Results are written under `experiments/.../inference/` as `nii.gz` (inverse HE applied by default when HE bins are configured).

### Evaluation

```bash
python eval_ddm2_local.py --patient_id 214878
# optional: --excel <xlsx> --batch 5 --gt_root <dir> --n2n_root <dir> --ddm2_root <dir> --output <xlsx>
```

Computes MAE / SSIM / LPIPS for the DDM2 output against Noisy and N2N references. Update the path constants at the top of `eval_ddm2_local.py` (and the argparse defaults) to your local layout before running.

## Data layout & important specifics

- **Self-supervision:** no clean CT ground truth is used for training; an N2N result serves as the teacher/conditioning image.
- **Data list:** an Excel file (`dataroot`) indexes cases; volumes live under `data_root`. Cases are grouped by `train_batches` / `val_batches` (e.g. `[5]`) and selected via `volume_mask`.
- **HU window:** `HU_MIN = -1000.0`, `HU_MAX = 2000.0` (in config).
- **Slice range:** `slice_range = [30, 80]`; `padding = 3`.
- **Histogram equalization:** enabled (`histogram_equalization: true`) using `bins_file` / `bins_mapped_file`; inference inverts HE to recover HU.
- **Diffusion:** `image_size = 512`, 1 channel, conditional; `n_timestep = 1000`, schedule `rev_warmup70`, `linear_start = 5e-5`, `linear_end = 0.01`.
- **Training:** diffusion `n_iter = 100000`, Adam `lr = 1e-4`, EMA decay `0.9999`; noise model `n_iter = 50000`. `stage2_file` must point to the `stage2_matched.txt` produced by Stage 2.

## Credits

Method: DDM2 (ICLR 2023). This repository adapts it to brain CT and is used here as a baseline.
