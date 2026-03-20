import importlib
import inspect
import pkgutil
import os

from .base import BaseDetector

_registry: dict[str, type] = {}


def _discover():
    pkg_dir = os.path.dirname(__file__)
    for _, module_name, _ in pkgutil.iter_modules([pkg_dir]):
        if module_name == "base":
            continue
        try:
            module = importlib.import_module(f"detectors.{module_name}")
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseDetector) and obj is not BaseDetector and obj.name:
                    _registry[obj.name] = obj
        except Exception as e:
            print(f"[detectors] failed to load '{module_name}': {e}")


_discover()


def list_detectors() -> list[dict]:
    return [{"id": name, "name": cls.name, "description": cls.description}
            for name, cls in _registry.items()]


def get_detector(name: str) -> BaseDetector | None:
    cls = _registry.get(name)
    return cls() if cls else None


def get_all_detectors() -> dict[str, BaseDetector]:
    return {name: cls() for name, cls in _registry.items()}
