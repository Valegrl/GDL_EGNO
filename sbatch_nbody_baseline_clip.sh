#!/bin/bash
#SBATCH --job-name=nbody_baseline_clip
#SBATCH --output=logs_sbatch/nbody_baseline_clip_%A_%a.out
#SBATCH --error=logs_sbatch/nbody_baseline_clip_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=p200981
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-5

SEEDS=(1 2 3 4 5)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID-1]}

echo "Running N-body baseline (lambda=0.0) with clipping for seed $SEED"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /home/users/u103229/GDL_EGNO

eval "$(conda shell.bash hook)"
conda activate /project/scratch/p200981/egno/conda_envs/egno

python main_simulation_simple_no.py \
  --config_by_file configs/nbody_baseline_clip_seed${SEED}.json \
  --outf /project/scratch/p200981/egno/logs/nbody_clip \
  --exp_name nbody_baseline_clip_seed${SEED}

echo "Training completed for seed $SEED (baseline with clipping)"