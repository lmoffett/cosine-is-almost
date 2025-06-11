import importlib
import logging
import pkgutil
from typing import Callable

log = logging.getLogger(__name__)


def load_train_dataloaders_method(package, module_name: str) -> Callable:
    """
    Dynamically gets a reference to a train_dataloaders method defined on a module, checking that the input typing is correct.
    """
    module_name = module_name.lower()

    # Iterate over all modules in the package
    for _, found_module_name, _ in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        if module_name in found_module_name.lower():
            try:
                # Dynamically import the module
                module = importlib.import_module(found_module_name)

                # Check if the module has the train_dataloaders method
                if hasattr(module, "train_dataloaders"):
                    method = getattr(module, "train_dataloaders")

                    # Check if the method has the correct signature
                    if callable(method):
                        # Verify the method signature
                        import inspect

                        signature = inspect.signature(method)
                        parameters = signature.parameters

                        if (
                            "data_path" in parameters
                            and "train_dir" in parameters
                            and "val_dir" in parameters
                            and "image_size" in parameters
                            and "batch_sizes" in parameters
                        ):
                            return method
                        else:
                            raise ValueError(
                                "train_dataloaders method missing required parameter(s)"
                            )

            except ImportError as e:
                print(f"Could not import module {found_module_name}: {e}")

    raise ImportError(
        f"Module {module_name} with 'train_dataloaders' method not found in package."
    )
