#!/usr/bin/env bash
#SBATCH --account=nesi99991
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --partition=hgx
#SBATCH --gpus-per-node=A100:1

# display information about the available GPUs
nvidia-smi

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# load required environment modules
module purge
module load Miniconda3/22.11.1-1

# activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda deactivate
conda activate /nesi/nobackup/nesi99991/introhpc2403/$USER/venv

# execute the script
python train_model.py "${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
