# Submission

The challenge is based on the [Sequoia](https://github.com/lebrice/Sequoia) library. In our framework you can develop CL methods in such a way that enables you to obtain a lot of results in different settings (dataset + set of assumptions).

As you can see in [DummyMethod](https://github.com/fgolemo/cl-workshop-cvpr21-docker/blob/main/submission/dummy_method.py), the minimal requirements for a method are:

- A `configure(self, setting: <Setting>)` function which is called once before training starts 
- A `fit(self, train_env: <Environment>)` function which is called once per task
- A `get_actions(self, observations: Observations, action_space: gym.Space) -> Actions` function which should always be ready to return a prediction

A great starting point to get a deeper understanding of the framework is the [example folder of Sequoia](https://github.com/lebrice/Sequoia/tree/master/examples).
