#!/bin/bash
#PBS -N ddos_inference
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=5:ngpus=1
#PBS -o logs/inference.log

cd $PBS_O_WORKDIR
mkdir -p logs
module load python/3.9.6
module load cuda
python src/inference_pipeline.py --config config/config.yaml --input data/processed/test_data.csv --output reports/inference_predictions.csv
