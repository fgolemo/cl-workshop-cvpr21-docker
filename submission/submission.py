""" You can modify this file and every file in this directory
but the get_method() functions must return the method you're planning to use for each
track.
"""
from sequoia import Method
from sequoia.settings import PassiveSetting, ActiveSetting, Setting

from submission.SL_examples import ExampleMethod
from submission.dummy_method import DummyMethod


def get_method_sl() -> Method[PassiveSetting]:
    """Returns the Method to be be used for the supervised learning track.

    Adjust this to your liking. You may create your own Method, or start from some of
    the provided example submissions.

    NOTE: Your Method can configure itself for the Setting it will be applied on, in its
    `configure` method.

    Returns
    -------
    Method[PassiveSetting]
        A Method applicable to continual supervised learning Settings.
    """
    # Demo of how to solve the SL task:
    # return ExampleMethod(hparams=ExampleMethod.HParams())
    # This is a dummy solution that returns random actions for every observation.
    return DummyMethod()

    # You could also use some of the available methods:
    from submission.SL_examples.multihead_classifier import ExampleTaskInferenceMethod
    return ExampleTaskInferenceMethod(
        hparams=ExampleTaskInferenceMethod.HParams(max_epochs_per_task=1)
    )

    from sequoia.methods.ewc_method import EwcMethod, EwcModel
    return EwcMethod(hparams=EwcModel.HParams())

    from sequoia.methods.experience_replay import ExperienceReplayMethod
    return ExperienceReplayMethod(buffer_capacity=500)


def get_method_rl() -> Method[ActiveSetting]:
    """Returns the Method to be be used for the reinforcement learning track.

    Adjust this to your liking. You could create your own Method, or start from some of
    the provided examples.

    NOTE: Your Method can configure itself for the Setting it will be applied on, in its
    `configure` method.

    Returns
    -------
    Method[ActiveSetting]
        A Method applicable to continual reinforcement learning settings.
    """
    # This is a dummy solution that returns random actions for every observation.
    return DummyMethod()

    # You could also use some of the available methods from SB3:
    from sequoia.methods.stable_baselines3_methods.dqn import DQNMethod, DQNModel

    return DQNMethod(hparams=DQNModel.HParams(verbose=1))

    from sequoia.methods.stable_baselines3_methods.ppo import PPOMethod, PPOModel

    return PPOMethod(hparams=PPOModel.HParams(verbose=1))

    from sequoia.methods.stable_baselines3_methods.a2c import A2CMethod, A2CModel

    return A2CMethod(hparams=A2CModel.HParams(verbose=1))


def get_method() -> Method[Setting]:
    """Returns the Method to be applied to both reinforcement and supervised Settings.

    NOTE: Your Method can configure itself for the Setting it will be applied on, in its
    `configure` method.

    Returns
    -------
    Method[Setting]
        A Method applicable to continual learning Settings, both in reinforcement or
        supervised learning.
    """
    # This is a dummy solution that returns random actions for every observation.
    return DummyMethod()
