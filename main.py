""" DO NOT MODIFY THIS FILE. THIS IS PART OF THE AUTOMATIC EVALUATION!
ONLY MODIFY THE SCRIPT(S) IN ./submission/!
"""
import argparse

### FOR PERFORMANCE REASONS, THE SEQUOIA IMPORTS ARE BELOW AFTER ARGPARSE

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="sl",
        const="sl",
        nargs="?",
        choices=["sl", "rl", "both"],
        help="train and eval supervised setting ('sl'), reinforcement setting ('rl'), or 'both' (default: %(default)s)",
    )
    args = parser.parse_args()


    from sequoia.methods import Method
    from sequoia.settings import ClassIncrementalSetting, Results, Setting, IncrementalRLSetting

    from setting_proxy import SettingProxy
    from submission.submission import get_method


    def upload_results(setting: Setting, method: Method, results: Results):
        print(results)
        raise NotImplementedError("TODO: Upload the results to evalai.")


    def run_track(setting, config: str, method: Method):
        setting = SettingProxy(setting, config)
        results = setting.apply(method)
        upload_results(setting, method, results)


    def run_sl(method: Method):
        return run_track(ClassIncrementalSetting, "sl_track.yaml", method)


    def run_rl(method: Method):
        return run_track(IncrementalRLSetting, "rl_track.yaml", method)


    if args.mode == "sl":
        print("=== RUNNING SL SETTING")
        run_sl(get_method())
    elif args.mode == "rl":
        print("=== RUNNING RL SETTING")
        run_rl(get_method())
    else:
        print("=== RUNNING BOTH SETTINGS")
        # TODO: this shouldn't generate 2 separate yaml files but only one with both results
        run_sl(get_method())
        run_rl(get_method())
