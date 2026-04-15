#!/bin/bash
#PBS -N ddos_data_pipeline
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=5:ngpus=1
#PBS -o logs/data_pipeline.log

cd $PBS_O_WORKDIR
mkdir -p logs
module load python/3.9.6
python src/data_pipeline.py --config config/config.yaml
