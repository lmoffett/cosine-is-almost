from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from swandb.weights_and_biases.sweeps import (
    WandBConfigExtended,
    create_pydantic_model_for_method,
    generate_sweep_configs,
    guard_sweep_runtime,
    load_function_from_path,
)


# Example functions to generate models from
def func_with_required_params(param1: int, param2: str):
    pass


def func_with_optional_params(param1: int, param2: str = "default"):
    pass


def func_with_mixed_params(param1: int, param2: str = "default", param3: float = 1.23):
    pass


def func_without_annotations(param1, param2="default"):
    pass


@pytest.mark.parametrize(
    "func, input_data, expected_data, expect_error",
    [
        # Test case 1: Function with required parameters
        (
            func_with_required_params,
            {"param1": 42, "param2": "hello"},
            {"param1": 42, "param2": "hello"},
            False,
        ),
        # Test case 2: Missing required parameter
        (
            func_with_required_params,
            {"param1": 42},
            None,
            True,
        ),
        # Test case 3: Function with optional parameters
        (
            func_with_optional_params,
            {"param1": 42},
            {"param1": 42, "param2": "default"},
            False,
        ),
        # Test case 4: Function with mixed parameters
        (
            func_with_mixed_params,
            {"param1": 42, "param2": "custom"},
            {"param1": 42, "param2": "custom", "param3": 1.23},
            False,
        ),
        # Test case 5: Function without type annotations
        (
            func_without_annotations,
            {"param1": 42, "param2": "custom"},
            {"param1": 42, "param2": "custom"},
            False,
        ),
        # Test case 6: Invalid type for a parameter
        (
            func_with_required_params,
            {"param1": "invalid", "param2": "hello"},
            None,
            True,
        ),
    ],
)
def test_create_pydantic_model(func, input_data, expected_data, expect_error):
    """Test the create_pydantic_model function."""
    Model = create_pydantic_model_for_method(func)

    if expect_error:
        with pytest.raises(ValidationError):
            Model(**input_data)
    else:
        model_instance = Model(**input_data)
        assert model_instance.model_dump() == expected_data


def test_load_function_from_script(temp_dir):
    """Test loading a function from a Python script."""
    # Create a temporary Python script
    script = temp_dir / "train.py"
    script.write_text(
        """
def train_method(epochs: int, batch_size: int = 32):
    pass
"""
    )

    # Load the function
    func = load_function_from_path(f"{script}::train_method")
    assert callable(func)

    # Create a Pydantic model from the function
    Model = create_pydantic_model_for_method(func)
    instance = Model(epochs=10)
    assert instance.model_dump() == {"epochs": 10, "batch_size": 32}


def test_invalid_script_path():
    """Test loading a function from a non-existent script."""
    with pytest.raises(ValueError, match="Python script not found"):
        load_function_from_path("non_existent.py::train_method")


def test_missing_function_in_script(temp_dir):
    """Test loading a non-existent function from a script."""
    # Create a temporary Python script
    script = temp_dir / "train.py"
    script.write_text(
        """
def other_method():
    pass
"""
    )

    with pytest.raises(ValueError, match="Function 'train_method' not found"):
        load_function_from_path(f"{script}::train_method")


def test_standard_python_module():
    """Test loading a function from a standard Python module."""
    func = load_function_from_path("math.sqrt")
    assert callable(func)
    assert func(4) == 2

    from math import sqrt

    assert func == sqrt


def generate_sweep_combinations(params: dict) -> List[dict]:
    """Helper function to generate combinations of parameters"""
    import itertools

    keys = params.keys()
    values = [params[key]["values"] for key in keys]
    combinations = []
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))
    return combinations


@pytest.fixture
def basic_sweep_config():
    return {
        "description": "Run a ProtoPNet Sweep",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "best_prototypes_embedded_accuracy"},
        "parameters": {"num_prototypes": {"values": [1, 5]}},
        "train_method": "training/iccv_cosine/iccv_protopnet.py::train",
        "hyper_sweep_parameters": {
            "backbone": {
                "values": ["resnet50[pretraining=inaturalist]", "densenet161"]
            },
            "activation_function": {"values": ["cosine"]},
        },
        "preflight_parameters": {"num_epochs": 5},
    }


def test_basic_config_generation():
    """Test basic configuration generation without preflight"""
    config = {
        "description": "Test Sweep",
        "method": "bayes",
        "parameters": {},
        "train_method": "test.py::train",
        "gpu_time_limit": "60m",
    }

    base_config, sweep_configs = generate_sweep_configs(config.copy(), "test", False)

    assert len(sweep_configs) == 1
    assert base_config["command"] == [
        "python",
        "-m",
        "swandb",
        "sweep",
        "train",
        "test.py::train",
        "${args_no_hyphens}",
        "--gpu-time-limit=60",
    ]
    assert sweep_configs[0][1]["name"] == "test"


def test_hyper_parameter_combinations(basic_sweep_config):
    """Test that all combinations of hyper parameters are generated"""
    base_config, sweep_configs = generate_sweep_configs(
        basic_sweep_config.copy(), "test", False
    )

    # Should have 2 backbone values * 1 activation_function value = 2 combinations
    assert len(sweep_configs) == 2

    # Check that each combination is unique
    combinations = set()
    for params, config in sweep_configs:
        params_tuple = tuple(params.items())
        combinations.add(params_tuple)

    assert len(combinations) == 2, combinations


def test_multiple_overrides_combinations(basic_sweep_config):
    """Test that all combinations of hyper parameters are generated"""
    base_config, sweep_configs = generate_sweep_configs(
        basic_sweep_config.copy(),
        "test",
        False,
        overrides={"backbone": {"vgg19", "resnet50"}},
    )

    # Should have 2 backbone values * 1 activation_function value = 2 combinations
    assert len(sweep_configs) == 2

    # Check that each combination is unique
    combinations = set()
    for params, config in sweep_configs:
        params_tuple = tuple(params.items())
        backbones = config["parameters"]["backbone"]["values"]
        assert len(backbones) == 1
        assert backbones[0] in {"vgg19", "resnet50"}
        combinations.add(params_tuple)

    assert len(combinations) == 2, combinations


def test_preflight_parameter_requirement(basic_sweep_config):
    """
    Test that:
    1. Every regular parameter must have a corresponding preflight value
    2. The preflight values are correctly applied
    3. Hyper sweep parameters are preserved for grid search
    """
    basic_sweep_config = basic_sweep_config.copy()

    # Set up regular parameters
    basic_sweep_config["parameters"] = {
        "learning_rate": {"values": [0.001, 0.01, 0.1]},
        "batch_size": {"values": [32, 64, 128]},
        "optimizer": {"values": ["adam", "sgd"]},
    }

    # Test that missing preflight parameters raises an error
    basic_sweep_config["preflight_parameters"] = {
        "learning_rate": 0.005,
        "optimizer": "rmsprop",
        # Deliberately missing batch_size
    }

    with pytest.raises(ValueError) as exc_info:
        generate_sweep_configs(basic_sweep_config, "test", preflight=True)
    assert "Missing preflight values for: {'batch_size'}" in str(exc_info.value)

    # Now provide all required preflight parameters
    basic_sweep_config["preflight_parameters"]["batch_size"] = 64

    # Add some hyper sweep parameters
    basic_sweep_config["hyper_sweep_parameters"] = {
        "model_size": {"values": ["small", "large"]},
    }

    _, sweep_configs = generate_sweep_configs(
        basic_sweep_config, "test", preflight=True
    )

    assert len(sweep_configs) == 1
    _, config = sweep_configs[0]

    # Verify all parameters have their preflight values
    assert config["parameters"]["learning_rate"]["values"] == [0.005]
    assert config["parameters"]["batch_size"]["values"] == [64]
    assert config["parameters"]["optimizer"]["values"] == ["rmsprop"]

    # Verify hyper_sweep parameters remain unchanged for grid search
    assert config["parameters"]["model_size"]["values"] == ["small", "large"]

    # Test that extra preflight parameters are allowed
    basic_sweep_config["preflight_parameters"]["extra_param"] = "test"
    _, sweep_configs = generate_sweep_configs(
        basic_sweep_config, "test", preflight=True
    )
    assert len(sweep_configs) == 1
    _, config = sweep_configs[0]
    assert config["parameters"]["extra_param"]["values"] == ["test"]


def test_single_value_parameters_in_preflight(basic_sweep_config):
    """
    Test that parameters with single values:
    1. Don't require preflight values
    2. Use their original value if not overridden
    3. Can still be overridden by preflight parameters
    """
    basic_sweep_config = basic_sweep_config.copy()

    # Set up mix of single and multi-value parameters
    basic_sweep_config["parameters"] = {
        "learning_rate": {"values": [0.001, 0.01]},  # Multi-value, needs preflight
        "batch_size": {"values": [32]},  # Single value, doesn't need preflight
        "optimizer": {"values": ["adam"]},  # Single value, will be overridden
        "epochs": {"values": [10, 20, 30]},  # Multi-value, needs preflight
    }

    # Provide preflight values only for multi-value parameters and one override
    basic_sweep_config["preflight_parameters"] = {
        "learning_rate": 0.005,
        "epochs": 15,
        "optimizer": "sgd",  # Overriding a single-value parameter
    }

    _, sweep_configs = generate_sweep_configs(
        basic_sweep_config, "test", preflight=True
    )

    assert len(sweep_configs) == 1
    _, config = sweep_configs[0]

    # Check that multi-value parameters got their preflight values
    assert config["parameters"]["learning_rate"]["values"] == [0.005]
    assert config["parameters"]["epochs"]["values"] == [15]

    # Check that single-value parameter kept its original value
    assert config["parameters"]["batch_size"]["values"] == [32]

    # Check that overridden single-value parameter got its new value
    assert config["parameters"]["optimizer"]["values"] == ["sgd"]

    # Test that missing preflight value for multi-value parameter raises error
    bad_config = basic_sweep_config.copy()
    bad_config["preflight_parameters"] = {"learning_rate": 0.005}  # Missing epochs

    with pytest.raises(ValueError) as exc_info:
        generate_sweep_configs(bad_config, "test", preflight=True)
    assert "Missing preflight values for: {'epochs'}" in str(exc_info.value)


def test_parameter_override_in_preflight(basic_sweep_config):
    """
    Test that:
    - Preflight parameters can't override hyper_sweep_parameters
    - Preflight parameters are properly added
    - Regular parameters are set to 'fixed'
    """
    basic_sweep_config = basic_sweep_config.copy()

    # Add regular parameters
    basic_sweep_config["parameters"]["num_epochs"] = {"values": [1, 100]}
    basic_sweep_config["parameters"]["batch_size"] = {"values": [32, 64]}

    # Add hyper sweep parameters
    basic_sweep_config["hyper_sweep_parameters"] = {
        "model_size": {"values": ["small", "medium"]},
        "dropout": {"values": [0.1, 0.2]},
    }

    # Test that preflight parameter can override regular parameter
    basic_sweep_config["preflight_parameters"] = {
        "num_epochs": 5,
        "batch_size": "fixed",
        "new_param": "test_value",
        "num_prototypes": 10,
    }

    _, sweep_configs = generate_sweep_configs(
        basic_sweep_config.copy(), "test", preflight=True
    )
    assert len(sweep_configs) == 1  # Single config with grid search
    _, config = sweep_configs[0]

    # Check regular parameters are fixed
    assert config["parameters"]["batch_size"]["values"] == ["fixed"]

    # Check preflight parameter override and addition
    assert config["parameters"]["num_epochs"]["values"] == [5]
    assert config["parameters"]["new_param"]["values"] == ["test_value"]

    # Check hyper_sweep_parameters are preserved
    assert set(config["parameters"]["model_size"]["values"]) == {"small", "medium"}
    assert set(config["parameters"]["dropout"]["values"]) == {0.1, 0.2}

    # Test that trying to override a hyper_sweep_parameter raises an error
    conflicting_config = basic_sweep_config.copy()
    conflicting_config["preflight_parameters"]["model_size"] = "large"

    with pytest.raises(ValueError) as exc_info:
        generate_sweep_configs(conflicting_config, "test", preflight=True)
    assert "Preflight parameters cannot override hyper_sweep parameters" in str(
        exc_info.value
    )


def test_empty_hyper_parameters():
    """Test handling of empty hyper_sweep_parameters"""
    config = {
        "description": "Test Sweep",
        "method": "bayes",
        "parameters": {},
        "train_method": "test.py::train",
        "hyper_sweep_parameters": {},
    }

    base_config, sweep_configs = generate_sweep_configs(config.copy(), "test", False)
    assert len(sweep_configs) == 1


def test_missing_hyper_parameters():
    """Test handling of missing hyper_sweep_parameters key"""
    config = {
        "description": "Test Sweep",
        "method": "bayes",
        "parameters": {},
        "train_method": "test.py::train",
    }

    base_config, sweep_configs = generate_sweep_configs(config.copy(), "test", False)
    assert len(sweep_configs) == 1


def test_sweep_name_generation(basic_sweep_config):
    """Test that sweep names are generated correctly"""
    _, sweep_configs = generate_sweep_configs(basic_sweep_config.copy(), "base", False)

    expected_names = [
        "base:back=resnet50[pret=inat]&actifunc=cosine",
        "base:back=densenet161&actifunc=cosine",
    ]
    for (params, config), expected_name in zip(sweep_configs, expected_names):
        assert config["name"] == expected_name


def create_base_config():
    """Helper function to create a valid base configuration"""
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "accuracy"},
        "parameters": {"learning_rate": {"values": [0.001, 0.01]}},
        "hyper_sweep_parameters": {
            "backbone": {"values": ["resnet50", "densenet121"]},
            "activation_function": {"values": ["cosine", "l2"]},
        },
        "preflight_parameters": {"param1": 1, "param2": "test"},
    }


def test_minimal_valid_config_with_train_method():
    """Test minimal valid configuration with train_method"""
    config = create_base_config()
    config["train_method"] = "training/train.py::train"

    validated = WandBConfigExtended.model_validate(config)
    assert validated.train_method == "training/train.py::train"
    assert validated.program is None
    assert validated.command is None


def test_minimal_valid_config_with_program_command():
    """Test minimal valid configuration with program and command"""
    config = create_base_config()
    config["program"] = "train.py"
    config["command"] = ["python", "train.py"]

    validated = WandBConfigExtended.model_validate(config)
    assert validated.program == "train.py"
    assert validated.command == ["python", "train.py"]
    assert validated.train_method is None


def test_optional_description():
    """Test that description is optional"""
    config = create_base_config()
    config["train_method"] = "training/train.py::train"

    # Without description
    validated1 = WandBConfigExtended.model_validate(config)
    assert validated1.description is None

    # With description
    config["description"] = "Test sweep"
    validated2 = WandBConfigExtended.model_validate(config)
    assert validated2.description == "Test sweep"


def test_invalid_missing_required_fields():
    """Test that required fields are enforced"""
    invalid_configs = [
        {},  # Empty config
        {"method": "bayes"},  # Missing most fields
        {
            "method": "bayes",
            "metric": {"goal": "maximize"},
            "parameters": {},
        },  # Missing hyper_sweep_parameters and preflight_parameters
    ]

    for config in invalid_configs:
        with pytest.raises(ValidationError):
            WandBConfigExtended.model_validate(config)


def test_invalid_train_method_with_program_command():
    """Test that train_method cannot be specified with program/command"""
    config = create_base_config()
    config["train_method"] = "training/train.py::train"
    config["program"] = "train.py"
    config["command"] = ["python", "train.py"]

    with pytest.raises(ValidationError) as exc_info:
        WandBConfigExtended.model_validate(config)
    assert (
        "If train_method is specified, program and command must not be specified"
        in str(exc_info.value)
    )


def test_invalid_program_without_command():
    """Test that program cannot be specified without command"""
    config = create_base_config()
    config["program"] = "train.py"

    with pytest.raises(ValidationError) as exc_info:
        WandBConfigExtended.model_validate(config)
    assert "program and command must be specified together" in str(exc_info.value)


def test_invalid_command_without_program():
    """Test that command cannot be specified without program"""
    config = create_base_config()
    config["command"] = ["python", "train.py"]

    with pytest.raises(ValidationError) as exc_info:
        WandBConfigExtended.model_validate(config)
    assert "program and command must be specified together" in str(exc_info.value)


def test_metric_accepts_any_structure():
    """Test that metric field accepts any valid structure"""
    config = create_base_config()
    config["train_method"] = "training/train.py::train"

    # Test different metric structures
    metric_variants = [
        {"goal": "maximize", "name": "accuracy"},
        {"custom_field": "value"},
        ["list", "of", "values"],
        "simple_string",
        42,
    ]

    for metric in metric_variants:
        config["metric"] = metric
        validated = WandBConfigExtended.model_validate(config)
        assert validated.metric == metric


def test_parameters_accepts_arbitrary_dict():
    """Test that parameters accepts arbitrary dictionary structure"""
    config = create_base_config()
    config["train_method"] = "training/train.py::train"

    # Test different parameters structures
    parameters_variants = [
        {},
        {"simple": "value"},
        {"nested": {"key": "value"}},
        {"list": [1, 2, 3]},
        {"complex": {"nested": {"list": [{"dict": "value"}]}}},
    ]

    for parameters in parameters_variants:
        config["parameters"] = parameters
        validated = WandBConfigExtended.model_validate(config)
        assert validated.parameters == parameters


def test_preflight_parameters_accepts_arbitrary_dict():
    """Test that preflight_parameters accepts arbitrary dictionary structure"""
    config = create_base_config()
    config["train_method"] = "training/train.py::train"

    # Test different preflight_parameters structures
    preflight_variants = [
        {},
        {"simple": "value"},
        {"nested": {"key": "value"}},
        {"list": [1, 2, 3]},
        {"complex": {"nested": {"list": [{"dict": "value"}]}}},
    ]

    for preflight_params in preflight_variants:
        config["preflight_parameters"] = preflight_params
        validated = WandBConfigExtended.model_validate(config)
        assert validated.preflight_parameters == preflight_params


def test_parameters_constraint():
    # Both None
    with pytest.raises(ValidationError) as exc_info:
        WandBConfigExtended.model_validate(
            {
                "method": "bayes",
                "metric": {"goal": "maximize"},
                "train_method": "test.py::train",
                "preflight_parameters": {},
            }
        )
    assert (
        "At least one of parameters or hyper_sweep_parameters must be non-empty"
        in str(exc_info.value)
    )

    # Both empty
    with pytest.raises(ValidationError) as exc_info:
        WandBConfigExtended.model_validate(
            {
                "method": "bayes",
                "metric": {"goal": "maximize"},
                "train_method": "test.py::train",
                "preflight_parameters": {},
            }
        )
    assert (
        "At least one of parameters or hyper_sweep_parameters must be non-empty"
        in str(exc_info.value)
    )

    # Valid with only parameters
    config = WandBConfigExtended.model_validate(
        {
            "method": "bayes",
            "metric": {"goal": "maximize"},
            "train_method": "test.py::train",
            "parameters": {"learning_rate": {"values": [0.1, 0.01]}},
            "preflight_parameters": {},
        }
    )
    assert config.parameters is not None
    assert config.hyper_sweep_parameters is None

    # Valid with only hyper_sweep_parameters
    config = WandBConfigExtended.model_validate(
        {
            "method": "bayes",
            "metric": {"goal": "maximize"},
            "train_method": "test.py::train",
            "hyper_sweep_parameters": {"backbone": {"values": ["value1", "value2"]}},
            "preflight_parameters": {},
        }
    )
    assert config.parameters is None
    assert config.hyper_sweep_parameters is not None

    # Valid with both
    config = WandBConfigExtended.model_validate(
        {
            "method": "bayes",
            "metric": {"goal": "maximize"},
            "train_method": "test.py::train",
            "parameters": {"learning_rate": {"values": [0.1, 0.01]}},
            "hyper_sweep_parameters": {"backbone": {"values": ["value1", "value2"]}},
            "preflight_parameters": {},
        }
    )
    assert config.parameters is not None
    assert config.hyper_sweep_parameters is not None


@pytest.fixture
def mock_sweep():
    sweep = Mock()
    sweep.runs = []
    return sweep


@pytest.fixture
def mock_wandb_api(mock_sweep):
    with patch("wandb.Api") as mock_api:
        api_instance = Mock()
        api_instance.sweep = Mock(return_value=mock_sweep)
        mock_api.return_value = api_instance
        yield mock_api


class TestGuardSweepRuntime:

    def test_finished_runs_with_runtime(self, mock_wandb_api, mock_sweep):
        run1, run2 = Mock(), Mock()
        run1.state = "finished"
        run1.summary = {"actual_runtime": 1200}  # 20 minutes
        run2.state = "finished"
        run2.summary = {"actual_runtime": 1800}  # 30 minutes
        mock_sweep.runs = [run1, run2]

        with patch("subprocess.run") as mock_subprocess:
            guard_sweep_runtime("entity/project/sweep123", 60)
            mock_subprocess.assert_not_called()  # 50 min used out of 60

        with pytest.raises(
            SystemExit, match="Total runtime exceeded, not starting a new run."
        ):
            with patch("subprocess.run") as mock_subprocess:
                guard_sweep_runtime("entity/project/sweep123", 40)
                mock_subprocess.assert_called_once_with(
                    ["wandb", "sweep", "--cancel", "entity/project/sweep123"]
                )

    def test_running_jobs(self, mock_wandb_api, mock_sweep):
        fixed_time = datetime(2024, 2, 7, 12, 0)

        # Create mock datetime class
        mock_datetime = Mock()
        mock_datetime.now.return_value = fixed_time
        mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp

        run = Mock()
        run.state = "running"
        run.start_time = datetime(2024, 2, 7, 11, 30).timestamp()  # 30 minutes ago
        mock_sweep.runs = [run]

        # patch the import on the module because we can't patch the module itself (it's a built-in)
        with patch("swandb.weights_and_biases.sweeps.datetime", mock_datetime):
            with patch("subprocess.run") as mock_subprocess:
                guard_sweep_runtime("entity/project/sweep123", 60)
                mock_subprocess.assert_not_called()

    def test_mixed_states(self, mock_wandb_api, mock_sweep):
        mock_datetime = Mock()
        fixed_time = datetime(2024, 2, 7, 12, 0)
        mock_datetime.now.return_value = fixed_time
        mock_datetime.fromtimestamp.side_effect = datetime.fromtimestamp

        run1, run2 = Mock(), Mock()
        run1.state = "finished"
        run1.summary = {"actual_runtime": 1800}  # 30 minutes

        run2.state = "running"
        run2.start_time = datetime(2024, 2, 7, 11, 20).timestamp()  # 40 minutes ago

        mock_sweep.runs = [run1, run2]

        # patch the import on the module because we can't patch the module itself (it's a built-in)
        with pytest.raises(
            SystemExit, match="Total runtime exceeded, not starting a new run."
        ):
            with patch("swandb.weights_and_biases.sweeps.datetime", mock_datetime):
                with patch("subprocess.run") as mock_subprocess:
                    guard_sweep_runtime("entity/project/sweep123", 60)
                    mock_subprocess.assert_called_once()  # 70 min total > 60 min limit

    def test_missing_runtime_info(self, mock_wandb_api, mock_sweep):
        run = Mock()
        run.state = "finished"
        run.summary = {}  # No runtime info
        mock_sweep.runs = [run]

        with patch("subprocess.run") as mock_subprocess:
            guard_sweep_runtime("entity/project/sweep123", 60)
            mock_subprocess.assert_not_called()
