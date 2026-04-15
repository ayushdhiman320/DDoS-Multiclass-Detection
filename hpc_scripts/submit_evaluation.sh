#!/bin/bash
#PBS -N ddos_model_evaluation
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=5:ngpus=0
#PBS -o logs/evaluation.log

cd $PBS_O_WORKDIR
mkdir -p logs
module load python/3.9.6
python src/model_evaluation.py --config config/config.yaml
