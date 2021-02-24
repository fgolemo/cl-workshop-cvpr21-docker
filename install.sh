echo "=== CREATING NEW CONDA ENV 'clcomp21'"

conda env create -f environment.yaml
conda activate clcomp21
pip install -r requirements.txt
# NOTE: Required step, as evalai causes a dependency conflict with `Sequoia`.
pip install --upgrade wandb
evalai set_token EVALAI_TOKEN
echo "=== DONE. Now run 'make sl' for running the sl track or 'make rl' for the RL track or 'make help'"
