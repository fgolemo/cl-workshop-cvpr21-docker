#!/bin/bash

echo "=== CREATING CONDA ENV 'clcomp21'"
# source `conda info --base`/bin/activate base
conda env create -f environment.yaml
# echo "`conda shell.bash hook`"
conda activate clcomp21

echo "==== Installing Python packages (this may take a while) "
pip install -r requirements.txt
# NOTE: Required step, as evalai causes a dependency conflict with `Sequoia`.
pip install --upgrade wandb


EVALAI_TOKEN=`cat evalai_token.txt`
if [ $EVALAI_TOKEN == "YOUR_TOKEN_HERE" ]; then
    echo "Skipping evalai sign-in. "
    echo "NOTE: You need to copy your EvalAI Token from https://eval.ai/web/profile and paste it inside evalai_token.txt file."
    exit 1
else
    echo "Signing in to evalai using provided token."
    evalai set_token "$EVALAI_TOKEN"
fi

echo "=== DONE. Now run 'make sl' for running the sl track or 'make rl' for the RL track or 'make help'"
