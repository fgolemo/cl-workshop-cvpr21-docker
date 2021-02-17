""" First track - Supervised Learning Setting (synbols dataset) """
import argparse

from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting, Results, Setting, IncrementalRLSetting

from setting_proxy import SettingProxy
from submission import get_method


def upload_results(setting: Setting, method: Method, results: Results):
    raise NotImplementedError("TODO: Upload the results to evalai.")


def sl_track(method: Method) -> None:
    setting = SettingProxy(ClassIncrementalSetting, "sl_track.yaml")

    results = setting.apply(method)

    upload_results(setting, method, results)


def rl_track(method: Method) -> None:
    setting = SettingProxy(IncrementalRLSetting, "rl_track.yaml")
    
    results = setting.apply(method)

    upload_results(setting, method, results)


if __name__ == "__main__":
    # setting = ClassIncrementalSetting(dataset="synbols", nb_tasks=12)
    # setting.save_yaml("sl_track.yaml")
    
    # setting = IncrementalRLSetting(dataset="monsterkong", nb_tasks=12)
    # setting.save_yaml("rl_track.yaml")
    
    method = get_method()
    sl_track(method)
