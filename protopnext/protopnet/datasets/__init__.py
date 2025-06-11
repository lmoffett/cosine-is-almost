import importlib
import logging
import os

from . import util

log = logging.getLogger(__name__)


def training_dataloaders(dataset_name: str, **kwargs):
    paths = [__package__]

    if "PPNXT_DATALOADERS_MODULES" in os.environ:
        paths.append(os.environ["PPNXT_DATALOADERS_MODULES"])

    for path in paths:
        package = importlib.import_module(path)

        try:
            method = util.load_train_dataloaders_method(
                package, module_name=dataset_name
            )(**kwargs)

            log.info(
                "Found dataloaders for %s in %s", dataset_name, package.__path__[0]
            )

            return method
        except ImportError:
            log.info("No dataloaders for %s in %s", dataset_name, package.__path__[0])

    return None
