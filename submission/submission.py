""" You can modify this file and every file in this directory 
but the get_method() functions must return the method you're planning to use for each
track.
"""
from sequoia import Method
from sequoia.settings import PassiveSetting, ActiveSetting, Setting

from submission.example_method import ExampleMethod
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
