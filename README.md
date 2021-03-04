# CVPR'21 Continual Learning Challenge Seed

See repo for CVPR'21 Continual Learning Challenge, both for the Supervised Learning and Reinforcement Learning track.

Join the community on the [ContinualAI slack](https://join.slack.com/t/continualai/shared_invite/enQtNjQxNDYwMzkxNzk0LTBhYjg2MjM0YTM2OWRkNDYzOGE0ZTIzNDQ0ZGMzNDE3ZGUxNTZmNmM1YzJiYzgwMTkyZDQxYTlkMTI3NzZkNjU) (the #cl-workshop channel)

## Prerequisites

- Install Docker (or Docker Desktop if you're on Mac/Win) and **make sure it's running**
- Make sure you can run the `docker` command without root. This is needed for the evalai submission tool. 
- Install Unix command line tools (like `rm`, and `make`) 
    - On Linux, you already have everything
    - On Mac, install `homebrew` and through homebrew, the `make` package
    - On Windows... idk, wipe your hard drive and install a proper OS? (Or try Cygwin or Windows Subsystem for Linux)
- Install Miniconda/Anaconda and have the conda/python/pip binaries for that available on your `$PATH` and in your shell.
- Create an account at http://eval.ai, create a participant team and click through the steps on the [challenge submission site](https://eval.ai/web/challenges/challenge-page/829/submission) (select your team, click next, agree to the terms & conditions) 

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