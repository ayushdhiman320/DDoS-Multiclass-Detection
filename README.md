# DDoS Multiclass Detection Pipeline

Production-ready end-to-end pipeline for CICDDoS2019 multiclass attack detection across 12 classes:
`LDAP, NetBIOS, MSSQL, UDP, UDPLag, SSDP, DNS, NTP, Syn, TFTP, Portmap, SNMP`.

## Project Structure

```
config/
  config.yaml
  model_configs/
src/
  data_pipeline.py
  feature_engineering.py
  model_training.py
  model_evaluation.py
  inference_pipeline.py
  utils/
hpc_scripts/
notebooks/
```

## Dataset Paths (Configured)

- `/home/richa.kumari/ayushminor/dataset/CSV-01-12/01-12`
- `/home/richa.kumari/ayushminor/dataset/CSV-03-11/03-11`

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate ddos-multiclass
```

## Pipeline Usage

```bash
python src/data_pipeline.py --config config/config.yaml
python src/feature_engineering.py --config config/config.yaml
python src/model_training.py --config config/config.yaml
python src/model_evaluation.py --config config/config.yaml
python src/inference_pipeline.py --config config/config.yaml --input data/processed/test_data.csv --output reports/inference_predictions.csv
```

## HPC Submission

```bash
qsub hpc_scripts/submit_data_pipeline.sh
qsub hpc_scripts/submit_training.sh
qsub hpc_scripts/submit_evaluation.sh
qsub hpc_scripts/submit_inference.sh
```

All PBS scripts use:
- Queue: `gpu`
- Python module: `python/3.9.6`
- `cd $PBS_O_WORKDIR`
- Logs under `logs/`

## Outputs

- Processed datasets: `data/processed/{train_data.csv,val_data.csv,test_data.csv,final_multiclass_data.csv}`
- Models: `models/*.joblib`
- Metrics and plots: `reports/` and `reports/figures/`
- Inference alerts: `reports/alerts.log`
