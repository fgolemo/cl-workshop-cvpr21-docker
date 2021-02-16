from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Dict, Generic, List, Type, TypeVar

from sequoia.common import Config
from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting, Results, Setting
from sequoia.settings.base import SettingABC

from env_proxy import EnvironmentProxy

logger = getLogger(__file__)

# IDEA: Dict that indicates for each setting, which attributes are *NOT* writeable.
_readonly_attributes: Dict[Type[Setting], List[str]] = {
    ClassIncrementalSetting: ["test_transforms"]
}
# IDEA: Dict that indicates for each setting, which attributes are *NOT* readable.
_hidden_attributes: Dict[Type[Setting], List[str]] = {
    ClassIncrementalSetting: ["test_class_order"]
}

SettingType = TypeVar("SettingType", bound=Setting)


class SettingProxy(SettingABC, Generic[SettingType]):
    """ Proxy for a Setting.
    
    TODO: Creating the Setting locally for now, but we'd spin-up or contact a gRPC
    service" that would have at least the following endpoints:

    - get_attribute(name: str) -> Any:
        returns the attribute from the setting, if that attribute can be read.
 
    - set_attribute(name: str, value: Any) -> bool:
        Sets the given attribute to the given value, if that is allowed. 

    - train_dataloader()
    """

    def __init__(self, setting_type: Type[SettingType], **setting_kwargs):
        self._setting_type = setting_type
        self._setting = setting_type(**setting_kwargs)
        super().__init__()

    def _is_readable(self, attribute: str) -> bool:
        return attribute not in _hidden_attributes[self.__dict__["_setting_type"]]

    def _is_writeable(self, attribute: str) -> bool:
        return attribute not in _readonly_attributes[self.__dict__["_setting_type"]]

    def __getattr__(self, name: str):
        # NOTE: This only ever gets called if the attribute was not found on the
        if self._is_readable(name):
            print(f"Accessing missing attribute {name} from the 'remote' setting.")
            return self.get_attribute(name)
        raise AttributeError(
            f"Attribute {name} is either not present on the setting, or not marked as "
            f"readable!"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # Weird pytorch-lightning stuff:
        logger.debug(f"__setattr__ called for attribute {name}")
        if name in {"_setting_type", "_setting"}:
            assert name not in self.__dict__, f"Can't change attribute {name}"
            object.__setattr__(self, name, value)
        elif self._is_writeable(name):
            logger.info(f"Setting attribute {name} on the 'remote' setting.")
            self.set_attribute(name, value)
        else:
            raise AttributeError(f"Attribute {name} is marked as read-only!")

    def apply(self, method: Method, config: Config = None) -> Results:
        # TODO: Figure this out:

        # IDEA: Maybe if we use the classmethod from the setting type, then it will call
        # the right attributes on `self`, which we can then implement below?
        # Run the Training loop (which is defined in IncrementalSetting).
        results = self._setting_type.apply(self, method, config=config)

        logger.info(f"Resulting objective of Test Loop: {results.objective}")
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

    def get_attribute(self, name: str) -> Any:
        return getattr(self._setting, name)

    def set_attribute(self, name: str, value: Any) -> None:
        return setattr(self._setting, name, value)

    def train_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:
        raise NotImplementedError("TODO")

    def valid_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:
        raise NotImplementedError("TODO")

    def test_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> EnvironmentProxy:
        raise NotImplementedError("TODO")
