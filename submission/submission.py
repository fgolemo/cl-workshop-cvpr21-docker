""" You can modify this file and every file in this directory 
but the get_method() function must return the method you're planning to use
"""

from sequoia import Method

from submission.classification_method import ExampleMethod
from submission.dummy_method import DummyMethod


def get_method() -> Method:

    # ADJUST THIS TO YOUR LIKING.

    # return ExampleMethod(hparams=ExampleMethod.HParams()) # demo of how to solve the SL task

    return DummyMethod()  # this is a dummy solution that returns random results
