#!/bin/bash
#SBATCH -J lstm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH -o filename_%j.out          # File to save standard output
#SBATCH -e filename_%j.err          # File to save standard error
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=6:00:00
#SBATCH --mem=32GB

# Load necessary modules (if required)
# module load python/3.x.x  # Uncomment and modify if needed

# Print GPU status and hostname (these will be logged in the output file)
nvidia-smi
hostname
echo "Works till here"
echo "Running Python"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Arrays for folds and inputs
folds=(1 2 3 4 5)
inputs=(12)

# Loop through both folds and inputs
for fold in "${folds[@]}"
do
  for input in "${inputs[@]}"
  do
    # Skip the iteration where fold is 1 and input is 6
    echo "Running Python with fold: $fold and input: $input"
    python3 ./GluNet_T1DEXI.py "$fold" "$input"
  done
done
