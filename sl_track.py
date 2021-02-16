""" First track - Supervised Learning Setting (synbols dataset) """
import argparse

from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting, Results, Setting

from setting_proxy import SettingProxy
from submission import get_method


def upload_results(setting: Setting, method: Method, results: Results):
    raise NotImplementedError("TODO: Upload the results to evalai.")


def sl_track(method: Method) -> None:
    setting: ClassIncrementalSetting = SettingProxy(ClassIncrementalSetting, dataset="synbols")

    results = setting.apply(method)

    upload_results(setting, method, results)


if __name__ == "__main__":
    method = get_method()
    sl_track(method)

