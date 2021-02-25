# cl-workshop-cvpr21-docker

This repo will serve as the seed repo for the challenge, containing a docker-compose.yaml file that spins up the method and environment containers, as well as two subfolders: method and environment ("setting"), each with their own docker file. Participants only make changes to the methods folder/container and then upload that to the challenge and we pull this image from evalai, and run it against our own environment docker.

## Setup

We assume you have `make` installed, as well as miniconda or anaconda on your PATH.

1. Get your API AUthentication token from the [EvalAI Website](https://eval.ai/web/profile).

2. From the root directory of the repository, run the following command:

```console
$ ./install.sh <EVALAI_TOKEN>
$ conda activate clcomp21
```

## creating you solution

develop a new solution in the `submission/` folder. 

You can draw inspiration form the `submission/dummy_method.py` which is a random predictor,
or from `submission/classification_method.py` which is a standard neural net classifier.

make sure the `submission/submission.py` script actually calls your method.

For more details on how to develop methods within our framework Sequoia and to understand how to leverage its funtionalities, algorithms and models, please check out the [codebase of the librairie](https://github.com/lebrice/Sequoia/).


## Running the tracks locally:

- Supervised Learning track:

```bash
make sl
```

- Reinforcement Learning track:

```bash
make rl
```

- "Combined" track:

```bash
make both
```


## Making a challenge submission

```bash
make [upload-sl|upload-rl|upload-both]
```
