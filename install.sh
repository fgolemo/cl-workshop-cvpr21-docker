#!/bin/bash
set -e
EVALAI_TOKEN=${1?"Usage: $0 <EVALAI_TOKEN>"}


echo "=== CREATING CONDA ENV 'clcomp21'"
eval "$(conda shell.bash hook)"
conda env create -f environment.yaml || true
conda activate clcomp21

echo "==== Installing Python packages (this may take a while) "
pip install -r requirements.txt
# NOTE: Required step, as evalai causes a dependency conflict with `Sequoia`.
pip install --upgrade wandb

evalai set-token ${EVALAI_TOKEN}

echo "=== DONE. Now run 'make sl' for running the sl track or 'make rl' for the RL track or 'make help'"
