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

from sequoia.client.setting_proxy import SettingProxy
from submission.submission import get_method, get_method_rl, get_method_sl


def run_track(method: Method, setting: Setting, yamlfile: str) -> Results:
    setting = SettingProxy(setting, yamlfile)
    results = setting.apply(method)
    print(f"Results summary:\n" f"{results.summary()}")
    print("=====================")
    print(results.to_log_dict())


def run_sl_track(method) -> ClassIncrementalSetting.Results:
    return run_track(method, ClassIncrementalSetting, "sl_track.yaml")


def run_rl_track(method) -> IncrementalRLSetting.Results:
    return run_track(method, IncrementalRLSetting, "rl_track.yaml")


def run_rl_and_sl_track():
    method = get_method()

    results_sl = run_sl_track(method)
    print(f"SL Results summary: {results_sl.summary()}")
    print("\n======= FINISHED SL, NOW RUNNING RL\n")

    results_rl = run_rl_track(method)
    print(f"RL Results summary: {results_rl.summary()}")
    print("=====================")

    print(results_sl.to_log_dict())
    print(results_rl.to_log_dict())


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
        run_sl_track(get_method_sl())

    elif args.mode == "rl":
        print("=== RUNNING RL TRACK")
        run_rl_track(get_method_rl())
    else:
        print("=== RUNNING BOTH SETTINGS")
        run_rl_and_sl_track()
