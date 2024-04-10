#!/bin/sh
#!/usr/bin/python


#SBATCH --job-name=g20-scicite
#SBATCH --gpus=1
#SBATCH --partition=medium

which python

pip install --no-index --upgrade pip
pip install -r requirements.txt


srun python ./train.py
