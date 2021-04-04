#!/bin/bash

echo "=== UPDATING REPO FROM GIT"
git pull

echo "=== UPDATING PIP PACKAGES"
eval "$(conda shell.bash hook)"
conda activate clcomp21
pip install --upgrade -r requirements.txt
pip install --upgrade wandb
