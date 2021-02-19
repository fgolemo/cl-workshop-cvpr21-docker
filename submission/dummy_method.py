import gym
import numpy as np
from sequoia import Method, ClassIncrementalSetting, PassiveEnvironment, Observations, Actions, Environment, Rewards


class DummyMethod(Method, target_setting=ClassIncrementalSetting):
    """ dummy method that does nothing and always returns 0    
    """

    def __init__(self):
        pass

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        pass

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        pass

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        return self.target_setting.Actions(action_space.sample())
