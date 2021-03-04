import gym
import numpy as np
import tqdm
from sequoia import (
    Actions,
    ClassIncrementalSetting,
    Environment,
    Method,
    Observations,
    PassiveEnvironment,
    Rewards,
)


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
        with tqdm.tqdm(train_env) as train_pbar:
            for i, (observations, rewards) in enumerate(train_pbar):
                batch_size = observations.x.shape[0]

                y_pred = train_env.action_space.sample()

                # If we're at the last batch, it might have a different size, so we give
                # only the required number of values.
                if y_pred.shape[0] != batch_size:
                    y_pred = y_pred[:batch_size]

                if rewards is None:
                    rewards = train_env.send(y_pred)

                # train as you usually would.

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        y_pred = action_space.sample()
        assert action_space.shape[0] == observations.x.shape[0]
        return self.target_setting.Actions(y_pred)
