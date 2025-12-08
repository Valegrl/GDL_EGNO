#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --account=p200981
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=egno_mocap_run
#SBATCH --output=logs_sbatch/egno_mocap_run_%A_%a.out
#SBATCH --error=logs_sbatch/egno_mocap_run_%A_%a.err

export WDIR="/project/scratch/p200981/egno"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Array Task ID     = $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate "$WDIR/conda_envs/egno"
cd ~/EGNO

CONFIGS=(
  "configs/mocap_run_seed1.json"
  "configs/mocap_run_seed2.json"
  "configs/mocap_run_seed3.json"
  "configs/mocap_run_seed4.json"
  "configs/mocap_run_seed5.json"
)

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID-1]}
echo "Running config: $CFG"
python -u main_mocap_no.py --config "$CFG"

echo "Job completed at $(date)"
