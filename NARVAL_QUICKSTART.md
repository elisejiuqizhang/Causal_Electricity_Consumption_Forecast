# Narval (Compute Canada) - Quick Start

## 📋 Files Created for Narval

### Environment Setup
- **`setup_narval_env.sh`** - Automated environment installation script

### SLURM Job Scripts (Multi-Region - 28 jobs each)
- **`job_scripts/narval_gru_multi_train.sh`**
- **`job_scripts/narval_tcn_multi_train.sh`**
- **`job_scripts/narval_patchtst_multi_train.sh`**

### SLURM Job Scripts (Single-Region - 36 jobs each)
- **`job_scripts/narval_gru_single_train.sh`**
- **`job_scripts/narval_tcn_single_train.sh`**
- **`job_scripts/narval_patchtst_single_train.sh`**

### Utilities
- **`job_scripts/narval_submit_all.sh`** - Batch submission script
- **`NARVAL_GUIDE.md`** - Complete usage guide

---

## 🚀 Quick Start (5 Steps)

### 1. Connect to Narval
```bash
ssh elise@narval.alliancecan.ca
```

### 2. Transfer Project
From your local machine:
```bash
rsync -avz --progress ~/path/to/Causal_Electricity_Consumption_Forecast/ \
    elise@narval.alliancecan.ca:~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast/
```

### 3. Setup Environment (on Narval)
```bash
cd ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast
bash setup_narval_env.sh
```
⏱️ Takes ~10-15 minutes

### 4. Update Paths
Edit ALL `narval_*.sh` files in `job_scripts/` directory:

**Line to update:**
```bash
PROJECT_DIR="/home/elise/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast"
```

**Verify your username and account:**
```bash
#SBATCH --account=def-bboulet  # Check with: sacctmgr show associations user=$USER
#SBATCH --mail-user=elise.zhang@mail.mcgill.ca
```

### 5. Submit Jobs
```bash
cd ~/projects/def-bboulet/elise/Causal_Electricity_Consumption_Forecast/job_scripts

# Test first with single job
bash narval_submit_all.sh test

# Then submit all
bash narval_submit_all.sh all
```

---

## 📊 Resource Specifications

### Per Job
- **1 GPU** (NVIDIA A100 40GB on Narval)
- **4 CPUs** per task
- **8GB memory** per CPU (32GB total)
- **2 days** time limit
- Uses **SLURM array jobs** for parallel execution

### Total Jobs
| Category | Jobs | Total GPU-hours (est.) |
|----------|------|------------------------|
| Multi-region (all) | 84 | ~4,032 |
| Single-region (all) | 108 | ~5,184 |
| **Grand Total** | **192** | **~9,216** |

---

## 🔍 Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### View Logs
```bash
# Real-time
tail -f ~/scratch/causal_forecast/logs/gru_multi_*.out

# Check for errors
grep -i error ~/scratch/causal_forecast/logs/*.err
```

### Job Efficiency
```bash
seff <JOB_ID>
```

### Cancel Jobs
```bash
scancel <JOB_ID>        # Cancel specific job
scancel -u $USER        # Cancel all your jobs
```

---

## 📁 Directory Structure on Narval

```
~/projects/def-bboulet/elise/
└── Causal_Electricity_Consumption_Forecast/
    ├── data/                         # Data files
    ├── exps/                         # Training scripts
    ├── utils/                        # Utility modules
    ├── job_scripts/                  # Job submission scripts
    ├── setup_narval_env.sh          # Environment setup
    └── NARVAL_GUIDE.md              # This guide

~/scratch/causal_forecast/
├── logs/                             # SLURM output logs
└── outputs/                          # Training outputs
    └── forecast/
        ├── multi_region/
        │   ├── gru_multi_train/
        │   ├── tcn_multi_train/
        │   └── patchtst_multi_train/
        └── per_region/
            ├── gru_single_train/
            ├── tcn_single_train/
            └── patchtst_single_train/
```

---

## ⚙️ Key Differences from Your Original Scripts

### 1. **Module Loading**
```bash
# Old (specific to your previous cluster)
module load python/3.9 cuda/11.4 scipy-stack

# New (Narval)
module load python/3.10 cuda/11.8 scipy-stack
```

### 2. **Conda Activation**
```bash
# Old
source /home/elise/elise_envs/miniconda3/bin/activate
conda activate multivarCM

# New
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate causal_forecast
```

### 3. **Path Structure**
```bash
# Old
/home/elise/elise_projects/...

# New
/home/elise/projects/def-bboulet/elise/...  # Code
/home/elise/scratch/causal_forecast/...     # Outputs/logs
```

### 4. **Array Jobs vs Parallel**
```bash
# Old (GNU Parallel across nodes)
#SBATCH --nodes=12
parallel --jobs 288 python script.py ::: $PARAMS

# New (SLURM array jobs)
#SBATCH --nodes=1
#SBATCH --array=0-27
# Each array task runs one parameter combination
```

### 5. **GPU Request**
```bash
# Old (implicit or partition-based)
# No explicit GPU request

# New (explicit)
#SBATCH --gres=gpu:1
```

---

## 🎯 Submission Options

```bash
# Test single job first
bash narval_submit_all.sh test

# Submit by category
bash narval_submit_all.sh multi      # Multi-region only (84 jobs)
bash narval_submit_all.sh single     # Single-region only (108 jobs)

# Submit by model
bash narval_submit_all.sh gru        # All GRU experiments (64 jobs)
bash narval_submit_all.sh tcn        # All TCN experiments (64 jobs)
bash narval_submit_all.sh patchtst   # All PatchTST experiments (64 jobs)

# Submit everything
bash narval_submit_all.sh all        # All 192 jobs
```

---

## 🐛 Common Issues

### Issue: "Module not found"
```bash
module spider python    # Find available versions
module spider cuda      # Find available CUDA versions
```

### Issue: "Conda command not found"
```bash
# Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
```

### Issue: "Out of memory"
Reduce batch size in job script:
```bash
--batch_size 32   # Instead of 64
```

Or increase memory:
```bash
#SBATCH --mem-per-cpu=16G   # Instead of 8G
```

### Issue: "Job pending for long time"
Check your priority:
```bash
squeue -u $USER --start
sacctmgr show associations user=$USER
```

---

## 📥 Downloading Results

From your local machine:
```bash
# Download all results
rsync -avz --progress \
    elise@narval.alliancecan.ca:~/scratch/causal_forecast/outputs/ \
    ./local_results/

# Download specific experiment
scp -r elise@narval.alliancecan.ca:~/scratch/causal_forecast/outputs/forecast/multi_region/gru_multi_train/ \
    ./results/gru_multi/
```

---

## 📚 Resources

- **Full Guide**: See `NARVAL_GUIDE.md` for detailed instructions
- **Compute Canada Docs**: https://docs.alliancecan.ca/
- **Narval Info**: https://docs.alliancecan.ca/wiki/Narval
- **Support**: support@tech.alliancecan.ca

---

## ✅ Pre-Submission Checklist

- [ ] Project transferred to Narval
- [ ] Environment installed (`setup_narval_env.sh`)
- [ ] Updated `PROJECT_DIR` in all `narval_*.sh` scripts
- [ ] Verified account with `sacctmgr show associations user=$USER`
- [ ] Updated email address in scripts
- [ ] Data files accessible (in project or scratch)
- [ ] Test job submitted and verified (`bash narval_submit_all.sh test`)
- [ ] Ready to submit full batch

---

**Good luck with your experiments! 🚀**
