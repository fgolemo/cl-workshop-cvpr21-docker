# CVPR'21 Continual Learning Challenge Seed

Seed repo for the [CVPR'21 Continual Learning Challenge](https://eval.ai/web/challenges/challenge-page/829/overview), both for the Supervised Learning and Reinforcement Learning tracks.


Join the community on the [ContinualAI slack](https://join.slack.com/t/continualai/shared_invite/enQtNjQxNDYwMzkxNzk0LTBhYjg2MjM0YTM2OWRkNDYzOGE0ZTIzNDQ0ZGMzNDE3ZGUxNTZmNmM1YzJiYzgwMTkyZDQxYTlkMTI3NzZkNjU) (the #cl-workshop channel)

## Prerequisites

- Install Docker (or Docker Desktop if you're on Mac/Win) and **make sure it's running**
- Make sure you can run the `docker` command without root. This is needed for the evalai submission tool. 
- Install Unix command line tools (like `rm`, and `make`) 
    - On Linux, you already have everything
    - On Mac, install `homebrew` and through homebrew, the `make` package
    - On Windows, not 100% sure, but try using Cygwin or Windows Subsystem for Linux. PowerShell alone is NOT sufficient.
- Install Miniconda/Anaconda and have the conda/python/pip binaries for that available on your `$PATH` and in your shell.
- Create an account at http://eval.ai, create a participant team and click through the steps on the [challenge submission site](https://eval.ai/web/challenges/challenge-page/829/submission) (select your team, click next, agree to the terms & conditions) 

## Setup

1. Get your API Authentication token from [you profile on the EvalAI Website](https://eval.ai/web/profile).

2. From the root directory of the repository, run the following commands:

```console
./install.sh <EVALAI_TOKEN>
conda activate clcomp21
```

## Creating Your Solution

Submissions should be contained in the `submission/` folder. You can draw inspiration from the following examples:
 - [submission/dummy_method.py](submission/dummy_method.py):
        Model-free example that outputs random predictions/actions. Applicable to all tracks (RL and SL).
- [SL Examples](submission/SL_examples)
    - [Simple Classifier](submission/SL_examples/classifier.py):
        Standard neural net classifier without any CL-related mechanism. Works in the SL track, but has very poor performance.

    - [CL Regularized Classifier](submission/SL_examples/regularization_example.py):
        Adds a simple CL regularization loss to the classifier above. Still exhibits poor performance.
    
    - [Multi-Head / Task Inference Classifier](submission/SL_examples/multihead_classifier.py):
        Performs multi-head prediction, and a simple form of task inference. Gets better results that the two previous methods.

    - (More to be added shortly)

- [RL Examples](submission/RL_examples) (coming soon!)
- ["Both" Examples](submission/both_examples) (coming soon!)

Make sure to change the contents of `submission/submission.py`, so that the various `get_method` actually return your method to use for each task (`get_method_sl` -> SL track, `get_method_rl` -> RL Track, `get_method` -> Both/Bonus track)

For more information on the various CL Settings and Methods available in Sequoia, please check out [the Sequoia repository](https://github.com/lebrice/Sequoia/).


## Updating Sequoia and dependencies

This is already part of the `install.sh` script, so you don't have to run this if you only just installed everything. This is best run once every few weeks or if you notice the readme changed or there was announcement in the challenge Slack.

```bash
make update
```


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

Pick the right option for the track that you want to submit to (...-sl/-rl/-both):

```bash
make [upload-sl|upload-rl|upload-both]
```

## FAQ

**Q: When I'm trying to make a submission, it says "Error: You have not participated in this challenge"**

A: You do actually have to participate in the challenge. Please see the last step of the "Prerequisites" section above. On eval.ai, create a participant team, go to the submission page of this challenge (https://eval.ai/web/challenges/challenge-page/829/submission), click on your team, click next, agree to the terms & conditions. Then you are officially a participant and can make submissions. To verify that this worked, please run `evalai challenges --participant` on a terminal and make sure that it has our challenge listed there (ID: 829, Title: "Challenge track of "Workshop on Continual Learning in Computer Vision 2021")

---

**Q: I'm getting an error a la `Connection aborted. FileNotFoundError(2, No such file or directory) `**

A: Make sure your current user has access rights to the docker cli without sudo. To test this, run `docker run hello-world` (without sudo obvs).