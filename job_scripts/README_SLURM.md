# SLURM Job Scripts Guide

This directory contains both GNU Parallel and SLURM versions of job submission scripts for running forecasting experiments.

## Directory Structure

```
job_scripts/
├── README_SLURM.md                    # This file
├── logs/                              # Created automatically by SLURM scripts
│
├── Multi-Region Scripts (GNU Parallel)
├── 1110gru_multi_train.sh             # GRU multi-region (parallel)
├── 1110tcn_multi_train.sh             # TCN multi-region (parallel)
├── 1110patchtst_multi_train.sh        # PatchTST multi-region (parallel)
│
├── Single-Region Scripts (GNU Parallel)
├── 1110gru_univ_train.sh              # GRU single-region (parallel)
├── 1110tcn_univ_train.sh              # TCN single-region (parallel)
├── 1110patchtst_univ_train.sh         # PatchTST single-region (parallel)
│
├── Multi-Region Scripts (SLURM)
├── slurm_gru_multi_train.sh           # GRU multi-region (SLURM)
├── slurm_tcn_multi_train.sh           # TCN multi-region (SLURM)
├── slurm_patchtst_multi_train.sh      # PatchTST multi-region (SLURM)
│
└── Single-Region Scripts (SLURM)
    ├── slurm_gru_single_train.sh      # GRU single-region (SLURM)
    ├── slurm_tcn_single_train.sh      # TCN single-region (SLURM)
    └── slurm_patchtst_single_train.sh # PatchTST single-region (SLURM)
```

## SLURM Scripts Overview

### Resource Allocation
All SLURM scripts request:
- **Time**: 48 hours
- **CPUs**: 4 per task
- **Memory**: 32GB
- **GPU**: 1 GPU per job
- **Array Jobs**: Parallel execution of multiple parameter combinations

### Multi-Region Scripts
- **Array size**: 0-27 (28 jobs total)
- **Combinations**: 7 region lists × 4 feature sets
- **Experiments**: 
  - Region lists: 1-7 (different city groupings)
  - Feature sets: F0, F1, F2, F3

### Single-Region Scripts
- **Array size**: 0-35 (36 jobs total)
- **Combinations**: 9 regions × 4 feature sets
- **Experiments**:
  - Regions: Toronto, Peel, Hamilton, Brantford, Waterloo, London, Oshawa, Kingston, Ottawa
  - Feature sets: F0, F1, F2, F3

## Usage

### 1. Submit a SLURM Job

```bash
# Navigate to project root
cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast

# Submit multi-region jobs
sbatch job_scripts/slurm_gru_multi_train.sh
sbatch job_scripts/slurm_tcn_multi_train.sh
sbatch job_scripts/slurm_patchtst_multi_train.sh

# Submit single-region jobs
sbatch job_scripts/slurm_gru_single_train.sh
sbatch job_scripts/slurm_tcn_single_train.sh
sbatch job_scripts/slurm_patchtst_single_train.sh
```

### 2. Submit Specific Array Indices

Run only specific experiments:

```bash
# Run only first 4 experiments (region list 1, all feature sets)
sbatch --array=0-3 job_scripts/slurm_gru_multi_train.sh

# Run only feature set F3 for all regions (indices 3,7,11,15,19,23,27)
sbatch --array=3,7,11,15,19,23,27 job_scripts/slurm_gru_multi_train.sh

# Run single region (e.g., Toronto with all feature sets: indices 0-3)
sbatch --array=0-3 job_scripts/slurm_gru_single_train.sh
```

### 3. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific array job
scancel <JOB_ID>_<ARRAY_INDEX>
```

### 4. Check Outputs

Logs are saved in `job_scripts/logs/`:
- **Standard output**: `logs/<model>_<type>_<JOB_ID>_<ARRAY_INDEX>.out`
- **Standard error**: `logs/<model>_<type>_<JOB_ID>_<ARRAY_INDEX>.err`

```bash
# View recent log
tail -f job_scripts/logs/gru_multi_*.out

# Check for errors
grep -i error job_scripts/logs/gru_multi_*.err

# List all logs for a specific model
ls -lh job_scripts/logs/gru_multi_*
```

## Array Index Mapping

### Multi-Region Scripts (28 jobs)
```
Array Index | Region List | Feature Set
------------|-------------|------------
0           | 1           | F0
1           | 1           | F1
2           | 1           | F2
3           | 1           | F3
4           | 2           | F0
...         | ...         | ...
27          | 7           | F3
```

Formula: `region_idx = array_id / 4`, `feature_idx = array_id % 4`

### Single-Region Scripts (36 jobs)
```
Array Index | Region      | Feature Set
------------|-------------|------------
0           | Toronto     | F0
1           | Toronto     | F1
2           | Toronto     | F2
3           | Toronto     | F3
4           | Peel        | F0
...         | ...         | ...
35          | Ottawa      | F3
```

Formula: `region_idx = array_id / 4`, `feature_idx = array_id % 4`

## Customization

### Modify Resource Requests

Edit the `#SBATCH` directives in the script:

```bash
#SBATCH --time=96:00:00          # Increase time to 96 hours
#SBATCH --mem=64G                # Increase memory to 64GB
#SBATCH --cpus-per-task=8        # Increase CPUs to 8
#SBATCH --gres=gpu:2             # Request 2 GPUs
#SBATCH --partition=gpu          # Specify partition
```

### Modify Hyperparameters

Edit the `python "$SCRIPT"` command in the script:

```bash
python "$SCRIPT" \
    --epochs 1000 \              # Change number of epochs
    --batch_size 128 \           # Change batch size
    --lr 0.0005 \                # Change learning rate
    --n_folds 10 \               # Change number of folds
    ...
```

### Add Email Notifications

Add to `#SBATCH` directives:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com
```

## Troubleshooting

### Job Fails Immediately
- Check logs in `job_scripts/logs/`
- Verify conda environment exists: `conda env list`
- Verify GPU availability: `sinfo --format="%20N %10c %10m %25f %10G"`

### Out of Memory
- Reduce `--batch_size` in the script
- Increase `#SBATCH --mem=` allocation

### Out of Time
- Increase `#SBATCH --time=` limit
- Reduce `--epochs` or `--n_folds`

### Wrong Conda Environment
- Check environment name in script matches your setup
- Modify: `conda activate scratch` → `conda activate your_env_name`

## Alternative: GNU Parallel Scripts

For local or non-SLURM systems, use the GNU parallel scripts:

```bash
bash job_scripts/1110gru_multi_train.sh
bash job_scripts/1110tcn_multi_train.sh
bash job_scripts/1110patchtst_multi_train.sh
```

These run 4 jobs in parallel using GNU parallel.

## Results Location

Training results are saved in:
```
outputs/forecast/
├── multi_region/
│   ├── gru_multi_train/region_list{1-7}/...
│   ├── tcn_multi_train/region_list{1-7}/...
│   └── patchtst_multi_train/region_list{1-7}/...
└── per_region/
    ├── gru_single_train/{region}/...
    ├── tcn_single_train/{region}/...
    └── patchtst_single_train/{region}/...
```

Each experiment saves:
- `best_model.pth` - Best model weights
- `test_predictions.csv` - Test set predictions
- `overall_results.txt` - Metrics summary
- `test_forecast_{region}.png` - Visualization plots
- `fold_{i}/` - Per-fold results

## Feature Sets

- **F0**: IESO electricity data only (baseline)
- **F1**: All weather features + IESO electricity
- **F2**: Non-causally selected weather features + IESO electricity
- **F3**: Causally selected weather features + IESO electricity

## Contact

For issues or questions, check the project repository or contact the maintainer.
