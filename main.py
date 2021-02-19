""" DO NOT MODIFY THIS FILE. THIS IS PART OF THE AUTOMATIC EVALUATION!
ONLY MODIFY THE SCRIPT(S) IN ./submission/!
"""
import argparse

from sequoia.methods import Method
from sequoia.settings import (
    ClassIncrementalSetting,
    IncrementalRLSetting,
    Results,
    Setting,
)

from setting_proxy import SettingProxy
from submission.submission import get_method, get_method_rl, get_method_sl


def run_sl_track():
    method = get_method_sl()

    setting = SettingProxy(ClassIncrementalSetting, "sl_track.yaml")
    results: ClassIncrementalSetting.Results = setting.apply(method)
    print(f"Results summary: {results.summary()}")
    results_json = results.dumps_json()
    # print(f"Results json: {results_json}") # VERY verbose, will flood your shell.

    raise NotImplementedError("TODO: Upload the results to evalai.")


def run_rl_track():
    method = get_method_rl()

    setting = SettingProxy(IncrementalRLSetting, "rl_track.yaml")
    results = setting.apply(method)
    print(f"Results summary: {results.summary()}")
    results_json = results.dumps_json()
    # print(f"Results json: {results_json}")

    raise NotImplementedError("TODO: Upload the results to evalai.")


def run_rl_and_sl_track():
    method = get_method()

    sl_setting = SettingProxy(ClassIncrementalSetting, "sl_track.yaml")
    sl_results = sl_setting.apply(method)
    print(f"SL Results summary: {sl_results.summary()}")

    rl_setting = SettingProxy(IncrementalRLSetting, "rl_track.yaml")
    rl_results = rl_setting.apply(method)
    print(f"RL Results summary: {rl_results.summary()}")

    # TODO: Figure out how we want to weight the different objectives, since in SL
    # its the average accuracy, while in RL its the average reward per episode. 
    weighted_objective = 0.5 * sl_results.objective +  0.01 * rl_results.objective
    
    raise NotImplementedError("TODO: Upload the results to evalai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=False,
        default="sl",
        choices=["sl", "rl", "both"],
        help=(
            "Competition track to run. supervised setting ('sl'), reinforcement "
            "learning setting ('rl'), or 'both' (default: %(default)s)"
        ),
    )
    args = parser.parse_args()

    if args.mode == "sl":
        print("=== RUNNING SL TRACK")
        run_sl_track()

    elif args.mode == "rl":
        print("=== RUNNING RL TRACK")
        run_rl_track()
    else:
        print("=== RUNNING BOTH SETTINGS")
        run_rl_and_sl_track()
