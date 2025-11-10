# Compute Canada Narval Setup and Usage Guide

This guide provides instructions for running the Causal Electricity Consumption Forecasting experiments on Compute Canada's Narval cluster.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Data Transfer](#data-transfer)
3. [Environment Installation](#environment-installation)
4. [Job Submission](#job-submission)
5. [Monitoring Jobs](#monitoring-jobs)
6. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### 1. Connect to Narval

```bash
ssh elise@narval.alliancecan.ca
```

### 2. Set Up Directory Structure

```bash
# Create project directory (use projects space for code)
mkdir -p ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast

# Create scratch directory for outputs and logs
mkdir -p ~/scratch/causal_forecast/logs
mkdir -p ~/scratch/causal_forecast/outputs

# Navigate to project directory
cd ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast
```

### 3. Storage Best Practices on Compute Canada

- **`~/projects/`**: For code, scripts, small config files (backed up, 1TB quota)
- **`~/scratch/`**: For data, outputs, logs (NOT backed up, large quota, auto-deleted after 60 days of no access)
- **`~/home/`**: For personal files, environments (backed up, 50GB quota)

---

## Data Transfer

### Transfer Project Files

From your local machine:

```bash
# Transfer entire project
scp -r /path/to/Causal_Electricity_Consumption_Forecast elise@narval.alliancecan.ca:~/projects/def-bboulet/elise/

# Or use rsync (better for large transfers)
rsync -avz --progress /path/to/Causal_Electricity_Consumption_Forecast/ \
    elise@narval.alliancecan.ca:~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast/
```

### Transfer Data Files

If data is large, use Globus or transfer to scratch:

```bash
# Transfer data to scratch
scp -r /path/to/data/* elise@narval.alliancecan.ca:~/scratch/causal_forecast/data/
```

### Update Data Paths in Scripts

After transfer, update the data directory path in your training scripts to point to scratch:

```python
# In your Python scripts, set:
DATA_DIR = os.path.join(os.environ.get('SCRATCH', ''), 'causal_forecast', 'data', 'ieso_era5')
```

Or keep data in project directory if it's not too large:

```python
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'ieso_era5')
```

---

## Environment Installation

### Run Setup Script

```bash
cd ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast

# Make setup script executable
chmod +x setup_narval_env.sh

# Run setup (this takes ~10-15 minutes)
bash setup_narval_env.sh
```

### Manual Installation (Alternative)

If the automatic script fails:

```bash
# Load modules
module load python/3.10 cuda/11.8 scipy-stack

# Create conda environment
conda create -n causal_forecast python=3.10 -y
conda activate causal_forecast

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
conda install numpy pandas scikit-learn matplotlib seaborn -y
pip install tensorboard joblib tqdm
```

### Verify Installation

```bash
module load python/3.10 cuda/11.8 scipy-stack
conda activate causal_forecast

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"
```

---

## Job Submission

### Update Paths in Job Scripts

Before submitting, update these paths in **ALL** `narval_*.sh` scripts:

1. **Email**: Change `elise.zhang@mail.mcgill.ca` to your email
2. **Account**: Verify `def-bboulet` is correct (check with `sacctmgr show associations user=$USER`)
3. **Project path**: Update `PROJECT_DIR` line:
   ```bash
   PROJECT_DIR="/home/elise/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast"
   ```
4. **Log path**: Update output/error paths if needed:
   ```bash
   #SBATCH --output=/home/elise/scratch/causal_forecast/logs/...
   ```

### Submit Jobs

Navigate to job scripts directory:

```bash
cd ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast/job_scripts
```

#### Submit Multi-Region Jobs (28 jobs each)

```bash
sbatch narval_gru_multi_train.sh
sbatch narval_tcn_multi_train.sh
sbatch narval_patchtst_multi_train.sh
```

#### Submit Single-Region Jobs (36 jobs each)

```bash
sbatch narval_gru_single_train.sh
sbatch narval_tcn_single_train.sh
sbatch narval_patchtst_single_train.sh
```

#### Submit Specific Array Indices

```bash
# Only run first region list with all feature sets (indices 0-3)
sbatch --array=0-3 narval_gru_multi_train.sh

# Only run feature set F3 for all regions
sbatch --array=3,7,11,15,19,23,27 narval_gru_multi_train.sh

# Run Toronto experiments only (indices 0-3)
sbatch --array=0-3 narval_gru_single_train.sh
```

#### Test Run (Single Job)

Before running all jobs, test with a single configuration:

```bash
# Run only array index 0 (Region 1, Feature F0)
sbatch --array=0 narval_gru_multi_train.sh

# Check logs
tail -f ~/scratch/causal_forecast/logs/gru_multi_*_0.out
```

---

## Monitoring Jobs

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View detailed info for a specific job
squeue -j <JOB_ID>

# View only running jobs
squeue -u $USER -t RUNNING

# View pending jobs
squeue -u $USER -t PENDING
```

### Check Job Priority and Wait Time

```bash
# Estimate wait time
squeue -u $USER --start

# Check account usage and priority
sacctmgr show associations user=$USER
```

### Monitor Job Progress

```bash
# Watch log file in real-time
tail -f ~/scratch/causal_forecast/logs/gru_multi_12345678_0.out

# Check last 50 lines of log
tail -n 50 ~/scratch/causal_forecast/logs/gru_multi_12345678_0.out

# Search for errors
grep -i error ~/scratch/causal_forecast/logs/*.err
```

### Check Resource Usage

```bash
# After job completes, get statistics
seff <JOB_ID>

# For array jobs
seff <JOB_ID>_<ARRAY_INDEX>
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific array task
scancel <JOB_ID>_<ARRAY_INDEX>

# Cancel range of array tasks
scancel <JOB_ID>_[0-10]
```

---

## Troubleshooting

### Job Fails Immediately

1. **Check logs**:
   ```bash
   cat ~/scratch/causal_forecast/logs/gru_multi_*_0.err
   ```

2. **Verify modules load**:
   ```bash
   module load python/3.10 cuda/11.8 scipy-stack
   module list
   ```

3. **Check conda environment**:
   ```bash
   conda env list
   source $HOME/miniconda3/etc/profile.d/conda.sh
   conda activate causal_forecast
   ```

### CUDA Not Available

```bash
# In job script, add diagnostic:
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

If CUDA is not available:
- Check `#SBATCH --gres=gpu:1` is present
- Verify GPU partition is available: `sinfo -p gpu`

### Out of Memory

**Symptoms**: Job killed with exit code 137 or OOM error

**Solutions**:
1. Reduce batch size in script: `--batch_size 32` (from 64)
2. Increase memory: `#SBATCH --mem-per-cpu=16G` (from 8G)
3. Request more CPUs: `#SBATCH --ntasks-per-node=8`

### Out of Time

**Symptoms**: Job killed at time limit

**Solutions**:
1. Increase time limit: `#SBATCH --time=3-00:00:00` (3 days)
2. Reduce epochs: `--epochs 300` (from 500)
3. Use checkpoint/resume functionality (if implemented)

### Module Not Found

```bash
# Check if module exists
module spider python
module spider cuda

# Load correct versions
module load python/3.10 cuda/11.8
```

### File Not Found Errors

Check paths:
```bash
# Verify project structure
ls -la ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast/

# Check data files
ls -la ~/scratch/causal_forecast/data/
```

### Permission Denied

```bash
# Make scripts executable
chmod +x ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast/job_scripts/narval_*.sh
```

---

## Resource Guidelines

### Requesting Appropriate Resources

Based on typical deep learning jobs:

| Resource | Single-Region | Multi-Region | Notes |
|----------|---------------|--------------|-------|
| GPUs | 1 | 1 | One GPU is usually sufficient |
| CPUs | 4 | 4 | For data loading |
| Memory/CPU | 8-16G | 8-16G | Adjust based on data size |
| Time | 1-2 days | 2-3 days | With early stopping |
| Batch Size | 64 | 32-64 | Adjust if OOM |

### Optimal Job Sizing

- **Don't over-request**: Wastes allocation and increases wait time
- **Use `seff`**: After first job, check actual usage and adjust
- **Array jobs**: More efficient than individual submissions

### Example Adjustment

If `seff` shows you only used 4GB memory:

```bash
# Reduce from 8G to 6G per CPU
#SBATCH --mem-per-cpu=6G
```

---

## Results Retrieval

### Download Results

From your local machine:

```bash
# Download all results
scp -r elise@narval.alliancecan.ca:~/scratch/causal_forecast/outputs/ ./local_results/

# Or use rsync
rsync -avz --progress \
    elise@narval.alliancecan.ca:~/scratch/causal_forecast/outputs/ \
    ./local_results/
```

### Download Specific Results

```bash
# Download specific experiment
scp -r elise@narval.alliancecan.ca:~/scratch/causal_forecast/outputs/forecast/multi_region/gru_multi_train/ ./results/
```

---

## Useful Commands Summary

```bash
# Submit job
sbatch narval_gru_multi_train.sh

# Check status
squeue -u $USER

# Monitor log
tail -f ~/scratch/causal_forecast/logs/*.out

# Cancel job
scancel <JOB_ID>

# Check efficiency
seff <JOB_ID>

# Account info
sacctmgr show associations user=$USER
```

---

## Additional Resources

- [Compute Canada Documentation](https://docs.alliancecan.ca/)
- [Narval Cluster](https://docs.alliancecan.ca/wiki/Narval)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [Best Practices for GPU Jobs](https://docs.alliancecan.ca/wiki/AI_and_Machine_Learning)

---

## Contact

For Compute Canada support: support@tech.alliancecan.ca
For project issues: elise.zhang@mail.mcgill.ca
