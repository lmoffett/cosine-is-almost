import importlib
import inspect
import logging
import os
from typing import (
    Any,
    Callable,
    Union,
    get_args,
    get_origin,
)

log = logging.getLogger(__name__)

from pydantic import BaseModel, Field

def create_pydantic_model_for_method(func):
    """
    Dynamically create a Pydantic model based on the function signature.
    """
    params = inspect.signature(func).parameters
    fields = {}
    annotations = {}
    original_annotations = {}

    for name, param in params.items():
        # Store original annotation
        if param.annotation != inspect.Parameter.empty:
            original_annotations[name] = param.annotation
            annotation = param.annotation
            # Handle Union types
            if get_origin(annotation) is Union:
                # Convert Union types to a simpler type - use the first type in the Union
                annotation = get_args(annotation)[0]
        else:
            annotation = Any
            original_annotations[name] = Any

        # Handle default values for optional parameters
        if param.default != inspect.Parameter.empty:
            fields[name] = Field(default=param.default)
        else:
            fields[name] = Field(default=...)

        annotations[name] = annotation

    # Dynamically create and return the Pydantic model
    return type(
        "LauncherArgumentsModel",
        (BaseModel,),
        {
            "__annotations__": annotations,
            "_original_annotations": original_annotations,  # Store original annotations
            "model_config": {'arbitrary_types_allowed': True},
            **fields,
        },
    )


def load_function_from_path(path: str) -> Callable:
    """
    Dynamically load a function from a Python script file or a Python module.

    Example inputs:
        - "path/to/script.py::train_method" for a Python file
        - "some.module.name.train_method" for a standard Python module

    Args:
        path (str): Path to the Python file with "::function_name" or module name.

    Returns:
        Callable: The loaded function.

    Raises:
        ValueError: If the input format is invalid or the function cannot be loaded.
    """
    if "::" in path:
        # Case 1: File path with "::function_name"
        script_path, func_name = path.split("::", 1)

        # Verify the script exists
        if not os.path.isfile(script_path):
            raise ValueError(f"Python script not found: {script_path}")

        # Load the script as a module
        module_name = os.path.splitext(os.path.basename(script_path))[
            0
        ]  # Use the filename as the module name
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Retrieve the function
        try:
            func = getattr(module, func_name)
        except AttributeError:
            raise ValueError(
                f"Function '{func_name}' not found in script '{script_path}'"
            )
    else:
        # Import the launcher dynamically
        try:
            module_path, func_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except (ValueError, ImportError, AttributeError):
            log.error(f"Error loading train method '{path}'. Is it in your PYTHONPATH?")
            raise

    return func
