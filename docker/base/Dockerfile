FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y git zsh nano xvfb freeglut3-dev pkg-config libfontconfig1-dev
# moved this up so it doesn't rerun on every build
RUN pip install --upgrade git+https://www.github.com/lebrice/Sequoia.git@cvpr_competition_dev#egg=sequoia[monsterkong] wandb