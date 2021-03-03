from typing import Tuple, Optional

import gym
import numpy as np
import torch
from contextlib import contextmanager
from gym import Space, spaces
from sequoia.settings.passive.cl import ClassIncrementalSetting
from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    Rewards,
)
from sequoia.common.spaces import Sparse
from torch import Tensor, nn
from torch.nn import functional as F
from submission.SL_examples.classifier import Classifier, ExampleMethod
from logging import getLogger

logger = getLogger(__file__)


class MultiHeadClassifier(Classifier):
    def __init__(
        self,
        observation_space: Space,
        action_space: spaces.Discrete,
        reward_space: spaces.Discrete,
    ):
        super().__init__(observation_space, action_space, reward_space)
        # Use one output layer per task, rather than a single layer.
        self.output_heads = nn.ModuleList()
        # Use the output layer created in the Classifier constructor for task 0.
        self.output_heads.append(self.output)

        # NOTE: The optimizer will be set here, so that we can add the parameters of any
        # new output heads to it later.
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def create_output_head(self) -> nn.Module:
        return nn.Linear(self.representations_size, self.n_classes).to(
            self.device
        )

    def get_or_create_output_head(self, task_id: int) -> nn.Module:
        """ Retrieves or creates a new output head for the given task index.
        
        Also adds its params to the optimizer.
        """
        task_output_head: nn.Module
        if len(self.output_heads) > task_id:
            task_output_head = self.output_heads[task_id]
        else:
            logger.info(f"Creating a new output head for task {task_id}.")
            task_output_head = self.create_output_head()
            self.output_heads.append(task_output_head)
            assert self.optimizer, "need to set `optimizer` on the model."
            self.optimizer.add_param_group({"params": task_output_head.parameters()})
        return task_output_head

    def forward(self, observations: Observations) -> Tensor:
        observations = observations.to(self.device)

        x = observations.x
        task_labels: Tensor = observations.task_labels

        # NOTE: Uncommenting this can be useful when debugging the task inference below:
        # if len(self.output_heads) > 2 and not self.training:
        #     secret = task_labels
        #     task_labels = None

        if task_labels is not None:
            # We have task labels.
            # NOTE: The batch might contain data from more than one task.
            unique_task_ids, inverse_indices = torch.unique(
                task_labels, return_inverse=True
            )
            unique_task_ids = unique_task_ids.tolist()

            if len(unique_task_ids) == 1:
                # All items come from the same task:
                task_id = unique_task_ids[0]
                # Switch the output head, and then do prediction as usual:
                self.output = self.get_or_create_output_head(task_id)
                return super().forward(observations)

            batch_size = len(task_labels)
            all_indices = torch.arange(batch_size, dtype=int, device=self.device)

            # Placeholder for the predicitons for each item in the batch.
            task_features = [None for _ in range(batch_size)]
            task_outputs = [None for _ in range(batch_size)]

            shared_features = self.encoder(x)

            for i, task_id in enumerate(unique_task_ids):
                ## Get the forward pass slice for this task.

                task_mask = inverse_indices == i
                # Indices of the batch elements that are from task `task_id`.
                task_indices = all_indices[task_mask]
                task_x = x[task_mask]

                # NOTE: In this example the encoder is shared across all tasks, but
                # you could also do a different encoding for each task.
                # task_encoder = self.encoders[task_id]

                ## Get the classifier for this task.
                task_output_head = self.get_or_create_output_head(task_id)

                # Here we reuse the shared features:
                # task_h_x = task_encoder(task_x)
                task_h_x = shared_features[task_mask]
                # Get the predictions:
                task_logits = task_output_head(task_h_x)

                # Store the outputs before they are stacked later.
                for i, index in enumerate(task_indices):
                    # task_features[index] = task_h_x[i]
                    task_outputs[index] = task_logits[i]

            ## 'Merge' the results.
            assert all(item is not None for item in task_outputs)
            logits = torch.stack(task_outputs)

            # NOTE: We could also merge the hidden vectors from each task:
            # assert all(item is not None for item in task_features)
            # features = torch.stack(task_features)

            return logits
        else:
            # We don't have access to task labels (`task_labels` is None).
            # --> Perform a simple kind of task inference:
            # 1. Perform a forward pass with each task's output head;
            # 2. Merge these predictions into a single prediction somehow.

            # NOTE: This assumes that the observations are batched.
            batch_size = x.shape[0]
            all_indices = torch.arange(batch_size, dtype=int, device=self.device)
            # NOTE: In this example the encoder is shared across all tasks, but
            # you could also do a different encoding for each task.
            shared_features = self.encoder(x)

            n_known_tasks = len(self.output_heads)
            # Tasks encountered previously and for which we have an output head.
            known_task_ids: list[int] = list(range(n_known_tasks))
            assert known_task_ids
            # Placeholder for the predictions from each output head for each item in the
            # batch
            task_features = [None for _ in known_task_ids]
            task_outputs = [None for _ in known_task_ids]

            for task_id in known_task_ids:
                ## Get the forward pass for this task.
                task_x = x
                ## Get the classifier for this task.
                task_output_head = self.get_or_create_output_head(task_id)
                # task_encoder = self.encoders[task_id]

                # task_h_x = task_encoder(task_x)
                task_h_x = shared_features

                task_logits = task_output_head(task_h_x)

                # task_features[task_id] = task_h_x
                task_outputs[task_id] = task_logits

            ## 'Merge' the results.
            assert all(item is not None for item in task_outputs)
            # Stack the predictions
            logits_from_each_output_head = torch.stack(
                task_outputs, dim=-1
            )  # [B, N, T]
            
            # Simple kind of task inference:
            # For each item in the batch, use the output head that has the highest logit
            max_logits_across_heads, max_index = logits_from_each_output_head.max(
                dim=-1
            )  # [B, N]
            logits = max_logits_across_heads
            return logits
            # y_pred = max_logits_across_heads.argmax(-1)
            # task_inference_acc = (secret == y_pred).int().sum().float() / batch_size
            # assert False, (y_pred, secret, task_inference_acc)
        return logits

    def on_task_switch(self, task_id: Optional[int]):
        """ Executed when the task switches (to either a known or unknown task).
        """
        if task_id is not None:
            # Switch the output head.
            self.output = self.get_or_create_output_head(task_id)

class ExampleTaskInferenceMethod(ExampleMethod):
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        self.model = MultiHeadClassifier(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # Share a reference to the Optimizer with the model, so it can add new weights
        # when needed.
        self.model.optimizer = self.optimizer

    def on_task_switch(self, task_id: Optional[int]):
        self.model.on_task_switch(task_id)

    def get_actions(self, observations, action_space):
        # FIXME: Debugging the task inference mechanism.
        # if self.testing and len(self.model.output_heads) == 5:
        #     actions = super().get_actions(observations, action_space)
        #     assert False, (actions, self.model._bobo)
        return super().get_actions(observations, action_space)
        
if __name__ == "__main__":
    from sequoia.settings.passive.cl import MultiTaskSetting, ClassIncrementalSetting

    # Multi-Task setting, used to give us the Upper bound performance of class-incremental algorithms.
    # setting = MultiTaskSetting(
    setting = ClassIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True,
    )
    method = ExampleTaskInferenceMethod()
    results = setting.apply(method)

    method = ExampleTaskInferenceMethod()

    # method.configure()
