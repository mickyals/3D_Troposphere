#!/bin/bash
# get tunneling info

### Jupyter Notebook on Slurm
# One terminal: 
# `srun --partition=gpunodes -c 1 --mem=4G --gres=gpu:rtx_2080:1 -t 60 --pty jupyter.sh`

# Another terminal:
# `ssh -N -L 8888:gpunode16:8888 poyuchen@comps0.cs.toronto.edu`

port=8888
node=$(hostname -s)
user=poyuchen

source venv/bin/activate
which python

# run jupyter notebook
jupyter-notebook --no-browser --port=${port} --ip=${node}