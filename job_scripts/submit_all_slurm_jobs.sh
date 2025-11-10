#!/bin/bash
# Utility script to submit all SLURM jobs for the forecasting experiments
# Usage: bash submit_all_slurm_jobs.sh [option]
# Options:
#   multi     - Submit only multi-region jobs
#   single    - Submit only single-region jobs
#   all       - Submit all jobs (default)
#   <model>   - Submit specific model (gru, tcn, patchtst)

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to job_scripts directory
cd "$(dirname "$0")"

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   SLURM Job Submission Utility${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Create logs directory
mkdir -p logs
echo -e "${GREEN}✓${NC} Logs directory ready: ./logs/"

# Function to submit a job
submit_job() {
    local script=$1
    local description=$2
    
    if [ -f "$script" ]; then
        echo -e "\n${YELLOW}Submitting:${NC} $description"
        echo -e "  Script: $script"
        
        job_id=$(sbatch "$script" | awk '{print $NF}')
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓ Submitted successfully!${NC} Job ID: $job_id"
        else
            echo -e "  ${RED}✗ Submission failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Script not found:${NC} $script"
        return 1
    fi
}

# Parse command line argument
OPTION=${1:-all}

case $OPTION in
    multi)
        echo -e "${YELLOW}Mode:${NC} Multi-region jobs only\n"
        submit_job "slurm_gru_multi_train.sh" "GRU Multi-Region (28 jobs)"
        submit_job "slurm_tcn_multi_train.sh" "TCN Multi-Region (28 jobs)"
        submit_job "slurm_patchtst_multi_train.sh" "PatchTST Multi-Region (28 jobs)"
        ;;
    
    single)
        echo -e "${YELLOW}Mode:${NC} Single-region jobs only\n"
        submit_job "slurm_gru_single_train.sh" "GRU Single-Region (36 jobs)"
        submit_job "slurm_tcn_single_train.sh" "TCN Single-Region (36 jobs)"
        submit_job "slurm_patchtst_single_train.sh" "PatchTST Single-Region (36 jobs)"
        ;;
    
    gru)
        echo -e "${YELLOW}Mode:${NC} GRU models only\n"
        submit_job "slurm_gru_multi_train.sh" "GRU Multi-Region (28 jobs)"
        submit_job "slurm_gru_single_train.sh" "GRU Single-Region (36 jobs)"
        ;;
    
    tcn)
        echo -e "${YELLOW}Mode:${NC} TCN models only\n"
        submit_job "slurm_tcn_multi_train.sh" "TCN Multi-Region (28 jobs)"
        submit_job "slurm_tcn_single_train.sh" "TCN Single-Region (36 jobs)"
        ;;
    
    patchtst)
        echo -e "${YELLOW}Mode:${NC} PatchTST models only\n"
        submit_job "slurm_patchtst_multi_train.sh" "PatchTST Multi-Region (28 jobs)"
        submit_job "slurm_patchtst_single_train.sh" "PatchTST Single-Region (36 jobs)"
        ;;
    
    all)
        echo -e "${YELLOW}Mode:${NC} All jobs\n"
        echo -e "${BLUE}--- Multi-Region Jobs ---${NC}"
        submit_job "slurm_gru_multi_train.sh" "GRU Multi-Region (28 jobs)"
        submit_job "slurm_tcn_multi_train.sh" "TCN Multi-Region (28 jobs)"
        submit_job "slurm_patchtst_multi_train.sh" "PatchTST Multi-Region (28 jobs)"
        
        echo -e "\n${BLUE}--- Single-Region Jobs ---${NC}"
        submit_job "slurm_gru_single_train.sh" "GRU Single-Region (36 jobs)"
        submit_job "slurm_tcn_single_train.sh" "TCN Single-Region (36 jobs)"
        submit_job "slurm_patchtst_single_train.sh" "PatchTST Single-Region (36 jobs)"
        ;;
    
    *)
        echo -e "${RED}Error:${NC} Unknown option '$OPTION'"
        echo ""
        echo "Usage: bash submit_all_slurm_jobs.sh [option]"
        echo ""
        echo "Options:"
        echo "  multi     - Submit only multi-region jobs (84 total)"
        echo "  single    - Submit only single-region jobs (108 total)"
        echo "  all       - Submit all jobs (192 total) [default]"
        echo "  gru       - Submit GRU model jobs only (64 total)"
        echo "  tcn       - Submit TCN model jobs only (64 total)"
        echo "  patchtst  - Submit PatchTST model jobs only (64 total)"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}✓ Job submission completed!${NC}"
echo ""
echo "Monitor your jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  ./logs/"
echo -e "${BLUE}==================================================${NC}"
