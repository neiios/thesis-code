# Program usage

1. Create a `.env` file in the root of the project based on the example file.

2. Run the following commands:

```sh
# Generate dataset
scala run ./src/generate.sc -- \
  --model openai/gpt-4.1 \
  --samples-per-request 20 \
  --snippets-per-topic 40 \
  --topics-file ./src/topics.json \
  --schema-file ./src/schema.json \
  --prompts-file ./src/prompts.yaml \
  --output-dir ./results

# Merged separate datasets
cat ./results/classes_for_data_snippets.jsonl  ./results/null_checks_snippets.jsonl  ./results/throws_snippets.jsonl > ./results/raw_full.jsonl

# Tokenize snippets
scala run ./src/tokenize.sc -- \
  --input ./results/raw_full.jsonl \
  --output-dir ./results

# Create dataset plots
uv run -m src.plot -i ./results/processed_full.jsonl -o ./results/plots

# Optimize CNN
uv run -m src.cnn.optimize -i ./results/processed_full.jsonl -o ./results/lstm_optimized --max-trials 20 --overwrite

# Optimize LSTM
uv run -m src.lstm.optimize -i ./results/processed_full.jsonl -o ./results/lstm_optimized --max-trials 20 --overwrite
```

# Connecting to MIF HPC

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

# Accessing files from HPC

```sh
# Use sshfs to mount the folder from hpc locally
sshfs user6969@hpc.mif.vu.lt:/scratch/lustre/home/user6969/thesis-code thesis-code-hpc
```
