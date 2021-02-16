# cl-workshop-cvpr21-docker

This repo will serve as the seed repo for the challenge, containing a docker-compose.yaml file that spins up the method and environment containers, as well as two subfolders: method and environment ("setting"), each with their own docker file. Participants only make changes to the methods folder/container and then upload that to the challenge and we pull this image from evalai, and run it against our own environment docker.

## Setup

TODO: Fix this to use docker?

```console
$ conda env create -f environment.yaml
$ conda activate cvpr_competition 
$ pip install -r requirements.txt
```

## Running the tracks locally:

- Supervised Learning track:

    ```console
    python sl_track.py
    ```

- Reinforcement Learning track:

    TODO

- "Combined" track:

    TODO


## Making a challenge submission

TODO