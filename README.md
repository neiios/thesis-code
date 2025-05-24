# Code Overview

## Technical Stack

- Python 3.12: Data analysis, visualization, and model training
- Keras 3 with PyTorch backend: LSTM and CNN model implementation
- scikit-learn: Data preprocessing, evaluation metrics, and ML utilities
- Scala 3: Code generation and tokenization
- uv: Python package management and dependency resolution
- Nix: Development environment management
- JSON/JSONL: Data storage and interchange format

## Directory Structure

```
./.
├── src/                              # Source code
│   ├── generate.sc                   # Scala script for dataset generation via LLM API
│   ├── tokenize.sc                   # Scala script for code tokenization
│   ├── build_cnn.py                  # CNN model architecture definition
│   ├── build_lstm.py                 # LSTM model architecture definition
│   ├── optimize_hyperparameters.py   # Hyperparameter optimization pipeline
│   ├── reader.py                     # Data loading and preprocessing utilities
│   ├── prompts.yaml                  # LLM prompt templates for code generation
│   ├── topics.json                   # Application domains for code generation
│   └── schema.json                   # API response schema definition
├── visualize_cnn.py                  # CNN visualization script
├── visualize_lstm.py                 # LSTM visualization script
├── pyproject.toml                    # Python project dependencies
└── flake.nix                         # Nix development environment
```

## Quick Start

### 0. Prerequisites

The program has several complex platform-specific dependencies. It is recommended to run on:
- Linux distribution (preferred)
- Windows Subsystem for Linux (WSL)

Development Environment Setup:
1. Install [Nix](https://nixos.org/download.html) for managing Scala, uv, and other tooling. Alternatively, install the required tools manually.
2. Create a `.env` file in the root directory based on the example file
3. Enter the Nix development shell and sync Python dependencies:

```sh
# Enter the Nix development environment
nix develop

# Sync Python dependencies with uv
uv sync
```

### 1. Generate Dataset
Generate Scala code snippets across multiple domains using LLM API:

```sh
scala run ./src/generate.sc -- \
  --model openai/gpt-4.1 \
  --samples-per-request 20 \
  --snippets-per-topic 40 \
  --topics-file ./src/topics.json \
  --schema-file ./src/schema.json \
  --prompts-file ./src/prompts.yaml \
  --output-dir ./results
```

### 2. Merge Datasets
Combine generated snippets from different categories:

```sh
cat ./results/classes_for_data_snippets.jsonl \
    ./results/null_checks_snippets.jsonl \
    ./results/throws_snippets.jsonl > ./results/raw_full.jsonl
```

### 3. Tokenize Code Snippets
Process and tokenize the generated code for ML training:

```sh
scala run ./src/tokenize.sc -- \
  --input ./results/raw_full.jsonl \
  --output-dir ./results
```

### 4. Visualize Dataset
Generate statistical plots and analysis of the dataset:

```sh
uv run -m src.plot_dataset -i ./results/processed_full.jsonl -o ./results/plots
```

### 5. Train Models

Optimize CNN hyperparameters:
```sh
uv run -m src.optimize_hyperparameters \
  -i ./results/processed_full.jsonl \
  -o ./results/cnn \
  --max-trials 30 \
  --epochs 10
```

Optimize LSTM hyperparameters:
```sh
uv run -m src.optimize_hyperparameters \
  -i ./results/processed_full.jsonl \
  -o ./results/lstm \
  --max-trials 30 \
  --epochs 10 \
  --lstm=true
```

### 6. Generate Final Visualizations
Create comprehensive analysis and visualizations:

```sh
uv run visualize_cnn.py
uv run visualize_lstm.py
```
## Results

Results and visualizations are available in the `results/` directory.

## Connecting to MIF HPC

```sh
# Connect to gpu
srun -p gpu --gres gpu --pty $SHELL

# Do undproductive things
wget http://launchpadlibrarian.net/589203768/libfakeroot_1.28-1ubuntu1_amd64.deb
dpkg-deb -R libfakeroot_1.28-1ubuntu1_amd64.deb libfakeroot_1.28
singularity build --sandbox /tmp/arch docker://archlinux
mkdir -pv /tmp/arch/scratch
fakeroot -l /scratch/lustre/home/$(whoami)/libfakeroot_1.28/usr/lib/x86_64-linux-gnu/libfakeroot/libfakeroot-sysv.so singularity shell -w /tmp/arch
pacman -Sy fastfetch && fastfetch

# Now install what you need to run your stuff and actually start being productive...
```

## Accessing files from HPC

```sh
# Use sshfs to mount the folder from hpc locally
sshfs user6969@hpc.mif.vu.lt:/scratch/lustre/home/user6969/thesis-code thesis-code-hpc
```
