""" Demo: Creates a simple new method and applies it to a single CL setting.
"""
from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

from collections import defaultdict
from pathlib import Path

import gym
import pandas as pd
import tqdm
import torch
from numpy import inf
from gym import spaces
from torch import Tensor, nn
from torchvision.models import resnet18, ResNet
from simple_parsing import ArgumentParser


# This "hack" is required so we can run `python examples/quick_demo.py`
sys.path.extend([".", ".."])
from sequoia import Method, Setting
from sequoia.common.hparams import HyperParameters, log_uniform
from sequoia.common.spaces import Image
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.passive.cl.objects import (
    Actions,
    Observations,
    PassiveEnvironment,
    Results,
    Rewards,
)


class ExampleModel(nn.Module):
    """ Simple classification model without any CL-related mechanism.

    To keep things simple, this demo model is designed for supervised
    (classification) settings where observations have shape [3, 28, 28] (ie the
    MNIST variants: Mnist, FashionMnist, RotatedMnist, EMnist, etc.)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
    ):
        super().__init__()
        image_space: Image = observation_space.x
        image_shape = observation_space[0].shape

        # This only works for classification / discrete action spaces.
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        n_classes = action_space.n

        self.encoder, self.representations_size = self.create_encoder(image_space)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.representations_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )
        self.loss = nn.CrossEntropyLoss()

    def create_encoder(self, image_space: Image) -> Tuple[nn.Module, int]:
        """ Create an encoder for the given image space. Returns the encoder, as well as
        the size of the resulting representations.
        """
        # TODO: THe input space here (from the quick_demo.py file in Sequoia) is based
        # on MNIST, whereas the SL tracks will use `synbols`, which has 228x228 images.
        if image_space.width == image_space.height == 28:
            # MNIST / Fashion-MNIST etc setup.
            encoder = nn.Sequential(
                nn.Conv2d(image_space.channels, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            features = 256
        elif image_space.width == image_space.height == 224:
            # Synbols dataset:
            resnet: ResNet = resnet18()
            features = resnet.fc.in_features 
            # Disable/Remove the last layer. 
            resnet.fc = nn.Sequential()
            encoder = resnet
        else:
            raise NotImplementedError(
                f"No encoder registered for the given image space {image_space}"
            )
        return encoder, features
        
    def forward(self, observations: Observations) -> Tensor:
        # NOTE: here we don't make use of the task labels.
        x = observations.x
        task_labels = observations.task_labels
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        return self.shared_step(batch, *args, **kwargs)

    def validation_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        return self.shared_step(batch, *args, **kwargs)

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        # Since we're training on a Passive environment, we get both
        # observations and rewards.
        observations: Observations = batch[0]
        rewards: Rewards = batch[1]
        image_labels = rewards.y

        # Get the predictions:
        logits = self(observations)
        y_pred = logits.argmax(-1)

        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy}
        return loss, metrics_dict


class ExampleMethod(Method, target_setting=ClassIncrementalSetting):
    """ Minimal example of a Method targetting the Class-Incremental CL setting.
    
    For a quick intro to dataclasses, see examples/dataclasses_example.py    
    """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-parameters of the demo model. """

        # Learning rate of the optimizer.
        learning_rate: float = log_uniform(1e-6, 1e-2, default=0.001)

    def __init__(self, hparams: HParams):
        self.hparams: ExampleMethod.HParams = hparams
        self.max_epochs: int = 1
        self.early_stop_patience: int = 2

        # We will create those when `configure` will be called, before training.
        self.model: ExampleModel
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        self.model = ExampleModel(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.
        
        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                postfix = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.training_step(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)

            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                postfix = {}
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.0

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.validation_step(batch)
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = valid_env
                best_epoch = i
            if i - best_epoch > self.early_stop_patience:
                print(f"Early stopping at epoch {i}.")

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.HParams, "hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = ""):
        """Creates an instance of this Method from the parsed arguments."""
        hparams: ExampleMethod.HParams = args.hparams
        return cls(hparams=hparams)


def get_method() -> Method:
    method = ExampleMethod(hparams=ExampleMethod.HParams())
    return method

