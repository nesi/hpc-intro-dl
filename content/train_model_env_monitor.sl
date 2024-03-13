#!/usr/bin/env bash
#SBATCH --account=nesi99991
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --partition=hgx
#SBATCH --gpus-per-node=A100:1

# monitor GPU usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,nounits -l 5 > "gpustats-${SLURM_JOB_ID}.csv" &

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# load required environment modules
module purge
module load TensorFlow/2.13.0-gimkl-2022a-Python-3.11.3

# execute the script
python train_model.py "${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
