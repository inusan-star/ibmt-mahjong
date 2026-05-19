# Intent-Belief Multi-task Transformer for Behavior Cloning in Mahjong

This repository provides a comprehensive pipeline for training and evaluating Mahjong AI models using Behavior Cloning from human expert data.

## Environment Setup

### 1. Create and Activate Environment

Use the provided `environment.yml` to create and activate the environment. This will install Python 3.9, project dependencies, and necessary tools.

```bash
conda env create -f environment.yml
conda activate ibmt-mahjong
```

### 2. Install mjx-convert

Download the `mjx-convert` repository from [this link](https://github.com/inusan-star/mjx-convert), then install the tool for log conversion. This version is a fork of the [mjx-convert](https://github.com/mjx-project/mjx-convert).

```bash
git clone -b mahjong --single-branch https://github.com/inusan-star/mjx-convert.git
cd mjx-convert
make install
pip install .
cd ..
```

## Data Preparation

### 1. Place Mjlog Files

Download Tenhou `.mjlog` files (e.g., from [Tenhou Log](https://tenhou.net/sc/raw/)) and place them into the `data/mjlogs/` directory. The `data/mjlog_list.json` file contains the list of `.mjlog` file identifiers used in this project.

```bash
mkdir -p data/mjlogs
# Place Tenhou .mjlog files in this directory
```

### 2. Convert Mjlogs to JSON

Convert the `.mjlog` files to JSON format using the `mjx-convert` tool. The converted JSON files will be generated in the `data/json_logs/` directory.

```bash
python src/data/convert_mjlog.py
```

### 3. Dataset Building

Convert JSON files into NumPy format (`.npy`). Processed data is saved in `data/processed/{mode}/{split}/`. Run for both `cnn` and `symbolic` modes.

```bash
python src/data/build_dataset.py --mode cnn
python src/data/build_dataset.py --mode symbolic
```

## Training

### 1. Model Training

Train the Mahjong AI models using the `trainer.py` script. Metrics are logged to Weights & Biases, and the best model is saved in `models/{arch}/{name}/`.

- `--arch`: Model architecture [`cnn`, `trans`, `mst`, `ibmt_intent`, `ibmt_belief`, `ibmt_full`]
- `--name`: Run name for WandB
- `--device`: Execution device (e.g., `cuda:0`, `cpu`)

```bash
python src/training/trainer.py --arch cnn --name exp_cnn --device cuda:0
python src/training/trainer.py --arch trans --name exp_trans --device cuda:0
python src/training/trainer.py --arch mst --name exp_mst --device cuda:0
python src/training/trainer.py --arch ibmt_intent --name exp_ibmt_intent --device cuda:0
python src/training/trainer.py --arch ibmt_belief --name exp_ibmt_belief --device cuda:0
python src/training/trainer.py --arch ibmt_full --name exp_ibmt_full --device cuda:0
```

## Evaluation

### 1. Action Evaluation

Evaluate the discard prediction performance using the `eval_action.py` script. This script calculates metrics including NLL, Top-1, and Top-3 Accuracy. Results are saved in `results/eval_action.csv`.

- `--runs`: List of run names to evaluate (multiple runs can be specified)
- `--device`: Execution device (e.g., `cuda:0`, `cpu`)

```bash
python src/evaluation/eval_action.py --runs exp_cnn exp_trans exp_mst exp_ibmt_intent exp_ibmt_belief exp_ibmt_full --device cuda:0
```

### 2. Intent Evaluation

Evaluate the agent's yaku prediction capability using the `eval_intent.py` script. This script calculates metrics including Exact Match Ratio, Micro F1-score, and Macro F1-score. Results are saved in `results/eval_intent.csv`.

- `--runs`: List of run names with intent heads to evaluate
- `--device`: Execution device (e.g., `cuda:0`, `cpu`)

```bash
python src/evaluation/eval_intent.py --runs exp_ibmt_intent exp_ibmt_full --device cuda:0
```

### 3. Belief Evaluation

Evaluate the accuracy of opponent shanten number prediction using the `eval_belief.py` script. This script calculates metrics including Top-1 Accuracy and Mean Absolute Error (MAE). Results are saved in `results/eval_belief.csv`.

- `--runs`: List of run names with belief heads to evaluate
- `--device`: Execution device (e.g., `cuda:0`, `cpu`)

```bash
python src/evaluation/eval_belief.py --runs exp_ibmt_belief exp_ibmt_full --device cuda:0
```
