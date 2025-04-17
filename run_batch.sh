#!/bin/bash
# Job name:
#SBATCH --job-name=kan
#
# Partition:
#SBATCH --partition=gpunodes
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=4

#SBATCH --mem=120G

#Number of GPUs, this should generally be in the form "gpu:A5000:[1-4] with the type included
#SBATCH --gres=gpu:rtx_a6000:1
#
# Wall clock limit:
#SBATCH --time=48:00:00

# Email address:
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dandandives@gmail.com
#
## Command(s) to run (example):
source venv/bin/activate
python main.py --config configs/config.yaml