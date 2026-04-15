#!/bin/bash
#PBS -N ddos_model_training
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=5:ngpus=1
#PBS -o logs/training.log

cd $PBS_O_WORKDIR
mkdir -p logs
module load python/3.9.6
module load cuda
python src/model_training.py --config config/config.yaml
