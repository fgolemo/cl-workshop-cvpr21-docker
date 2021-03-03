# CVPR'21 Continual Learning Challenge Seed

See repo for CVPR'21 Continual Learning Challenge, both for the Supervised Learning and Reinforcement Learning track.

Join the community on the [ContinualAI slack](https://join.slack.com/t/continualai/shared_invite/enQtNjQxNDYwMzkxNzk0LTBhYjg2MjM0YTM2OWRkNDYzOGE0ZTIzNDQ0ZGMzNDE3ZGUxNTZmNmM1YzJiYzgwMTkyZDQxYTlkMTI3NzZkNjU) (the #cl-workshop channel)

## Prerequisites

- Install Docker (or Docker Desktop if you're on Mac/Win) and **make sure it's running**
- Install Unix command line tools (like `rm`, and `make`) 
    - On Linux, you already have everything
    - On Mac, install `homebrew` and through homebrew, the `make` package
    - On Windows... idk, wipe your hard drive and install a proper OS? (Or try Cygwin or Windows Subsystem for Linux)
- Install Miniconda/Anaconda and have the conda/python/pip binaries for that available on your `$PATH` and in your shell.


## Setup

1. Get your API Authentication token from [you profile on the EvalAI Website](https://eval.ai/web/profile),
   and copy it into the `evalai_token.txt` file.

2. From the root directory of the repository, run the `./install.sh` script.

```console
./install.sh
conda activate clcomp21
```

## Creating Your Solution

Submissions should be contained in the `submission/` folder. You can draw inspiration from the following examples:
- [submission/example_method.py](submission/example_method.py):
    Standard neural net classifier without any CL-related mechanism. Works in the SL
    track, but has very poor performance.

- [submission/example_reg_method.py](submission/example_reg_method.py):
    Adds a simple regularization loss to the example SL method above. Still exhibits
    poor performance.

- [submission/dummy_method.py](submission/dummy_method.py):
    Model-free example that outputs random predictions/actions. Applicable to all tracks (RL and SL).

- (More to be added shortly)

Make sure to change the contents of `submission/submission.py`, so that the various `get_method` actually return your method to use for each task (`get_method_sl` -> SL track, `get_method_rl` -> RL Track, `get_method` -> Both/Bonus track)

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
