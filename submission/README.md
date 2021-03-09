# Submission

The challenge is based on the [Sequoia](https://github.com/lebrice/Sequoia) library. In our framework you can develop CL methods in such a way that enables you to obtain a lot of results in different settings (dataset + set of assumptions).

As you can see in [DummyMethod](dummy_method.py), the minimal requirements for a method are:

- A `configure(self, setting: <Setting>)` function which is called once before training starts 
- A `fit(self, train_env: <Environment>)` function which is called once per task
- A `get_actions(self, observations: Observations, action_space: gym.Space) -> Actions` function which should always be ready to return a prediction

A great starting point to get a deeper understanding of the framework is the [example folder of Sequoia](https://github.com/lebrice/Sequoia/tree/master/examples).


## Running the examples (no docker needed)

To run/debug one of the examples on Settings from Sequoia (including the one used for the SL track of the competition), run any of the examples by directly invoking them like so:
```console
$ python submission/SL_examples/classifier.py
```

You can also pass command-line arguments, like so
```console
$ python submission/SL_examples/multihead_classifier.py --max_epochs_per_task 10 --learning_rate 3e-4
```

To get a list of available options, use the `--help` option:
```console
$ python submission/SL_examples/regularization_example.py --help
usage: regularization_example.py [-h] [--learning_rate float] [--weight_decay float] [--max_epochs_per_task int]
                                 [--early_stop_patience int] [--reg_coefficient float] [--reg_p_norm int]

optional arguments:
  -h, --help            show this help message and exit

ExampleRegMethod.HParams ['hparams']:
   Hyperparameters of this improved method.

          Adds the hyper-parameters related the 'ewc-like' regularization to those of the
          ExampleMethod.

          NOTE: These `uniform()` and `log_uniform` and `HyperParameters` are just there
          to make it easier to run HPO sweeps for your Method, which isn't required for
          the competition.


  --learning_rate float
                        Learning rate of the optimizer. (default: 0.001)
  --weight_decay float  L2 regularization coefficient. (default: 1e-06)
  --max_epochs_per_task int
                        Maximum number of training epochs per task. (default: 10)
  --early_stop_patience int
                        Number of epochs with increasing validation loss after which we stop training. (default: 2)
  --reg_coefficient float
                        Coefficient of the ewc-like loss. (default: 1.0)
  --reg_p_norm int      Distance norm used in the regularization loss. (default: 2)
```


## Using a Model/Method in a submission:

1. Follow the installation installation instructions, including the docker setup.
2. Modify `get_method_SL()` in `submission/submission.py`, so that it returns an
   instance of your method of choice, rather than `DummyMethod`.
3. Running the SL track locally:
    ```console
    make sl
    ```
4. Making a real submission to the Challenge:
    ```console
    make upload-sl
    ```