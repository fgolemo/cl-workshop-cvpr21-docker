# cl-workshop-cvpr21-docker

This repo will serve as the seed repo for the challenge, containing a docker-compose.yaml file that spins up the method and environment containers, as well as two subfolders: method and environment ("setting"), each with their own docker file. Participants only make changes to the methods folder/container and then upload that to the challenge and we pull this image from evalai, and run it against our own environment docker.

## Setup

We assume you have `make` installed, as well as miniconda or anaconda on your PATH.

Cd into this directory and then
  
```bash
make install
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

```bash
make [upload-sl|upload-rl|upload-both]
```