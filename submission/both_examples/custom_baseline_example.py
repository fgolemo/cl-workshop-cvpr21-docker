from dataclasses import dataclass
from typing import Optional
import gym
from sequoia.common import Config, TrainerConfig
from sequoia.methods.baseline_method import BaselineMethod, BaselineModel
from sequoia.settings import (
    Actions,
    Environment,
    Observations,
    Rewards,
    Setting,
    SettingType,
)


class MyCustomModel(BaselineModel):
    @dataclass
    class HParams(BaselineModel.HParams):
        """ Hyper-Parameters of our customized version of the BaselineModel. """

        # Add your hparams here:

    def __init__(
        self, setting: Setting, hparams: "MyCustomModel.HParams", config: Config
    ):
        super().__init__(setting, hparams, config)

    def create_output_head(self, setting: Setting, task_id: int):
        return super().create_output_head(setting, task_id)

    def encode(self, observations):
        return super().encode(observations)

    def forward(self, observations: Observations):
        return super().forward(observations)

    def training_step(self, batch, batch_idx: int, **kwargs):
        return super().training_step(batch, batch_idx, **kwargs)

    def validation_step(self, batch, batch_idx: int, **kwargs):
        return super().validation_step(batch, batch_idx, **kwargs)

    def shared_step(
        self,
        batch,
        batch_idx,
        environment,
        loss_name,
        dataloader_idx=None,
        optimizer_idx=None,
    ):
        return super().shared_step(
            batch,
            batch_idx,
            environment,
            loss_name,
            dataloader_idx=dataloader_idx,
            optimizer_idx=optimizer_idx,
        )


class MyCustomMethod(BaselineMethod):
    def __init__(
        self,
        hparams: BaselineModel.HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        **kwargs,
    ):
        """ Creates an instance of thsi Method using the provided configuration options.

        Parameters
        ----------
        hparams : BaselineModel.HParams, optional
            Hyper-parameters of the BaselineModel used by this Method. Defaults to None.

        config : Config, optional
            Configuration dataclass with options like log_dir, device, etc. Defaults to
            None.

        trainer_options : TrainerConfig, optional
            Dataclass which holds all the options for creating the `pl.Trainer` which
            will be used for training. Defaults to None.

        **kwargs :
            If any of the above arguments are left as `None`, then they will be created
            using any appropriate value from `kwargs`, if present.
        """
        super().__init__(
            hparams=hparams, config=config, trainer_options=trainer_options, **kwargs
        )

    def create_model(self, setting: SettingType) -> BaselineModel[SettingType]:
        """Creates the BaselineModel (a LightningModule) for the given Setting.

        You could extend this to customize which model is used depending on the
        setting.

        TODO: As @oleksost pointed out, this might allow the creation of weird
        'frankenstein' methods that are super-specific to each setting, without
        really having anything in common.

        Args:
            setting (SettingType): An experimental setting.

        Returns:
            BaselineModel[SettingType]: The BaselineModel that is to be applied
            to that setting.
        """
        # Create the model, passing the setting, hparams and config.
        return MyCustomModel(setting=setting, hparams=self.hparams, config=self.config)

    def fit(
        self,
        train_env: Environment[Observations, Actions, Rewards],
        valid_env: Environment[Observations, Actions, Rewards],
    ):
        """Called by the Setting to train the method.
        Could be called more than once before training is 'over', for instance
        when training on a series of tasks.
        Overwrite this to customize training.
        """
        super().fit(train_env, valid_env)

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (actions) for a batch of observations.

        This gets called by the Setting during the test loop.
        """
        return super().get_actions(observations, action_space)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called when switching between tasks.

        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        self.model.on_task_switch(task_id)


if __name__ == "__main__":
    from sequoia.settings.active import IncrementalRLSetting
    from sequoia.settings.passive import ClassIncrementalSetting

    # You can use this Method on any Setting in the tree, including both the SL and RL
    # tracks of the competition.
    setting = IncrementalRLSetting.load_benchmark("rl_track.yaml")
    setting = ClassIncrementalSetting.load_benchmark("sl_track.yaml")

    method = MyCustomMethod()
    results = setting.apply(method)
    print(results.summary())
