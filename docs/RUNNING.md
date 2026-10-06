# Running the development experiment

This repository contains the active training and evaluation code and aggregate results. It does not include video clips, eye-tracking recordings, recovered event annotations, or model weights. Rebuilding the exact cohort requires those project-specific records. Downloading the source dataset alone is insufficient.

## Environment and cluster use

The supported environment is Python 3.12.11, Torch 2.9.0+cu128, and torchvision 0.24.0+cu128, with the packages in `requirements.txt`. The launcher snapshots code and records the commit, working changes, inputs, and configuration for each job.

On LRZ, use Slurm for installation, downloads, hashing, tests, video decoding, plotting, and model computation. Use login nodes for editing, inspecting files, and submitting jobs. Store datasets, models, caches, and outputs in project DSS storage. Use only the CPUs, memory, GPUs, and wall time assigned to the job. Downloads require an allocation with network access; model jobs use local weights in offline mode.

```bash
export ROUTEB_STORAGE=/path/to/project-storage
export ROUTEB_INITIALIZATION=/path/to/preserved-gaze-initialization
mkdir -p logs
sbatch slurm/fovlanding.sbatch setup
sbatch slurm/fovlanding.sbatch check
```

The default launcher requests four CPUs and 16 GB for 30 minutes. Installation may need a longer wall time. GPU modes require overriding those defaults. The recorded development jobs requested one H100, four CPUs, and 64 GB host memory:

```bash
sbatch --partition=lrz-hgx-h100-94x4 --qos=gpu --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=12:00:00 \
  slurm/fovlanding.sbatch train "$INPUTS" d4
```

Partitions, QoS, and account access depend on the cluster. Check the local allocation before submitting; this example is an LRZ request, not a portable scheduler configuration.

## Models

`prepare_weights.py` downloads and verifies the pinned InternVL base and official gaze adapter inside Slurm. `bootstrap_environment.sh` prepares the pinned upstream checkout. Run weight preparation only after accepting the upstream terms and providing sufficient project storage.

The preserved gaze initialization must contain processor/tokenizer files, `static/adapter_config.json`, `static/adapter_model.safetensors`, and `initialization.json`. The manifest must identify `representation: frozen_static_lora` and the local `source_base`. This historical name describes the stored initialization. During task adaptation, its single gaze adapter is trainable; the base model stays frozen.

The repository expects that preserved initialization; it does not recreate the earlier preservation and verification step automatically. Training explicitly uses `$ROUTEB_STORAGE/models/InternVL3_5-8B-HF`. For smoke checks, update `source_base` when relocating the initialization. Keep its source hashes and adapter bytes intact.

## Data contracts

The original source layout is `source/revision-1/Clips-small` and `source/revision-1/GazeData` under project storage. The development audit additionally requires:

- A preparation JSON with `standardized_trials.path` and its `sha256`.
- A dataset audit JSON with `trial_manifest.sha256` and `recovered_record_hashes`.
- An audited-trial JSONL with raw filenames and hashes.
- Recovered `mapping_handoff/original/fixed-splits.json`, `alignment-check.json`, and `confirmation_events/development-events.json`.
- The exact detector parameter source identified by the experiment provenance.

These records are private project inputs. Audit records contain paths; when relocating them, regenerate their path references and verify the referenced bytes. The audit verifies event hashes, raw trial hashes, timestamps, display assumptions, and decoded video PTS. It produces a development configuration and a visual inspection sample. Its success does not independently validate playback origin or trial calibration.

```bash
export ROUTEB_RECOVERED=/path/to/recovered-provenance
sbatch slurm/fovlanding.sbatch audit --development \
  --preparation /path/to/preparation.json \
  --dataset-audit /path/to/dataset-audit.json \
  --raw-manifest /path/to/audited-trials.jsonl
```

The checked-in target configuration is a template and deliberately lacks a project commit and completed audit. It must not be presented as a validated data configuration. Set its detector parameter path to the recovered source before auditing.

Use the completed audit to build paired targets and materialize causal frames:

```bash
sbatch slurm/fovlanding.sbatch build "$AUDIT"
sbatch slurm/fovlanding.sbatch export --audit "$AUDIT" --targets "$TARGETS"
```

Here `AUDIT` is the job's `audit` directory and `TARGETS` is the subsequent job's `targets` directory. The export produces an `inputs` directory with manifests, images, a training histogram, and a hash-pinned `scan.json`. Set `INPUTS` to that directory. The same target IDs and split are required across all conditions. Training records the initial unadapted model's validation scores before updating its adapter.

## Training and evaluation

Run the three DeepGaze conditions separately using the same input export and initialization. The default training configuration uses seed 0. Use `replicate` for the other seeds, or provide a configuration with the desired `seed` through `--config`. Request GPU resources for training and smoke checks.

```bash
sbatch slurm/fovlanding.sbatch train "$INPUTS" d1
sbatch slurm/fovlanding.sbatch train "$INPUTS" r4
sbatch slurm/fovlanding.sbatch train "$INPUTS" d4
sbatch slurm/fovlanding.sbatch ar-weights
sbatch --time=02:00:00 slurm/fovlanding.sbatch ar "$INPUTS" --device cpu --seed 0
```

Training outputs are written under `$ROUTEB_STORAGE/runs/MODE-JOBID/training`. The launcher supports continuation using `resume ORIGINAL_TRAINING INPUTS`, and matched new seeds using `replicate ORIGINAL_TRAINING INPUTS SEED`; these reuse the original code snapshot. A completed historical run is not silently updated to the current source.

For each seed, compare its matched D1, R4, and D4 runs:

```bash
sbatch slurm/fovlanding.sbatch compare --outputs "$INPUTS" \
  --run d1=/path/to/d1/training --run r4=/path/to/r4/training \
  --run d4=/path/to/d4/training
sbatch slurm/fovlanding.sbatch aggregate "$INPUTS" \
  /path/to/seed0/comparison-job /path/to/seed1/comparison-job \
  /path/to/seed2/comparison-job
sbatch slurm/fovlanding.sbatch baseline --outputs "$INPUTS" \
  --history-baselines --neural-ar neural_ar_d4=/path/to/ar/training
```

The comparison-job directories contain `best/` and `final/`. All runs must have matching input provenance. Spatial and linear baselines are fitted on training data; visual AR chooses its best epoch on validation data. Film-macro IG must subtract the same training KDE likelihood from every predictor. Older training logs instead report improvement over a histogram and cannot be used directly as the public benchmark.

`scripts/plot_results.py` reads the committed aggregate summary to regenerate the figure. Its optional `--refresh-from` is a release helper for the recorded completed run IDs; it is not a general collector for newly trained runs.
