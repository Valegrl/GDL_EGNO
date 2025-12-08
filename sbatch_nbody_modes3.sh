#!/bin/bash
#SBATCH --job-name=nbody_modes3
#SBATCH --output=logs_sbatch/nbody_modes3_%A_%a.out
#SBATCH --error=logs_sbatch/nbody_modes3_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=p200981
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-5

# Seeds array
SEEDS=(1 2 3 4 5)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID-1]}

echo "Running N-body with num_modes=3 for seed $SEED"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

cd /home/users/u103225/EGNO

# Activate conda environment
source /home/users/u103225/miniconda3/etc/profile.d/conda.sh
conda activate egno

# Run training with num_modes=3
python main_simulation_simple_no.py \
    --config_by_file configs/nbody_modes3_seed${SEED}.json

echo "Training completed for seed $SEED with num_modes=3"
