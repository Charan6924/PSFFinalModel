#!/bin/bash
#SBATCH --job-name=kernel_train
#SBATCH --account=dlw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --constraint=gpu2h100
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Load modules (adjust as needed for your cluster)
module purge
module load python/3.10
module load cuda/12.1

# Activate virtual environment if you have one
# source ~/venv/bin/activate

# Navigate to code directory
cd $SLURM_SUBMIT_DIR

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Run training
python FullTrainLoop.py

echo "End time: $(date)"
