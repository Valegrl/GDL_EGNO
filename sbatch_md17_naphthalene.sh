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
#SBATCH --job-name=md17_naphthalene
#SBATCH --output=logs_sbatch/md17_naphthalene_%A_%a.out
#SBATCH --error=logs_sbatch/md17_naphthalene_%A_%a.err

export WDIR="/project/scratch/p200981/egno"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Array Task ID     = $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate "$WDIR/conda_envs/egno"
cd ~/EGNO

CONFIGS=(
  "md17_naphthalene_seed1.json"
  "md17_naphthalene_seed2.json"
  "md17_naphthalene_seed3.json"
  "md17_naphthalene_seed4.json"
  "md17_naphthalene_seed5.json"
)

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID-1]}
echo "Running config: $CFG"
python -u main_md17_no.py --config_by_file --config "$CFG"

echo "Job completed at $(date)"
