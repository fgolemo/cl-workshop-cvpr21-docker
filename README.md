# CVPR'21 Continual Learning Challenge Seed

See repo for CVPR'21 Continual Learning Challenge, both for the Supervised Learning and Reinforcement Learning track.

## Prerequisites

- Install Docker (or Docker Desktop if you're on Mac/Win) and **make sure it's running**
- Install Unix command line tools (like `rm`, and `make`) 
    - On Linux, you already have them
    - On Mac, install `homebrew` and the make package
    - On Windows... idk, wipe your hard drive and install a proper OS? (Or try Cygwin or Windows Subsystem for Linux)
- Install Miniconda/Anaconda and have the conda/python/pip binaries for that available on your `$PATH` and in your shell.


## Setup

1. Get your API Authentication token from the [EvalAI Website in your profile](https://eval.ai/web/profile).

2. From the root directory of the repository, run the following command:

```console
./install.sh <EVALAI_TOKEN>
conda activate clcomp21
```

## Creating Your Solution

Develop a new solution in the `submission/` folder. 

You can draw inspiration from `submission/classification_method.py` which is a standard neural net classifier, which kinda works in the SL setting. The `submission/dummy_target.py` is a model-free example that outputs random predictions/actions but doesn't learn anything.

Make sure the `submission/submission.py` script actually calls your method.

For more details on how to develop methods within our framework and to understand how to leverage its functionalities, algorithms, and models, please check out [Sequoia, the library that this is all based on](https://github.com/lebrice/Sequoia/).


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
