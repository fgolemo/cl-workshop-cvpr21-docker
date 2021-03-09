FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y git zsh nano xvfb freeglut3-dev pkg-config libfontconfig1-dev
# moved this up so it doesn't rerun on every build
RUN pip install --upgrade git+https://www.github.com/lebrice/Sequoia.git@cvpr_competition_dev#egg=sequoia[monsterkong] wandb
WORKDIR /app
COPY ./build/ .
# just making sure we got everything
#RUN pip install --upgrade -r requirements.txt # NO, this breaks wandb
CMD xvfb-run --server-args="-screen 0 800x600x24+32" python ./main.py --mode sl


# to build: docker build -t sequoia-seed .
# to run locally: docker run -it --rm sequoia-seed