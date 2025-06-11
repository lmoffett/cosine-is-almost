from unittest.mock import Mock

import pytest

from protopnet.train.scheduling.scheduling import (
    TrainLayersUsingProtoPNetNames,
    TrainOptimizerLayers,
)


def test_train_layers_using_protopnet_names_warning():
    """Test that initialization raises deprecation warning"""
    with pytest.warns(
        DeprecationWarning, match="provided only for backwards compatibility"
    ):
        TrainLayersUsingProtoPNetNames(
            train_backbone=True,
            train_add_on_layers=True,
            train_prototype_layer=True,
            train_prototype_prediction_head=True,
        )


class TestTrainLayersUsingProtoPNetNames:
    @pytest.fixture
    def model(self):
        model = Mock()
        # Create mock parameters that simulate named_parameters() return value
        parameters = {
            "backbone.weight": Mock(requires_grad=False),
            "backbone.bias": Mock(requires_grad=False),
            "add_on_layers.weight": Mock(requires_grad=False),
            "add_on_layers.bias": Mock(requires_grad=False),
            "prototype_layer.weight": Mock(requires_grad=False),
            "prototype_layer.prototypes": Mock(requires_grad=False),
            "prototype_prediction_head.weight": Mock(requires_grad=False),
            "prototype_prediction_head.bias": Mock(requires_grad=False),
            "conv_offset.weight": Mock(requires_grad=False),
            "conv_offset.bias": Mock(requires_grad=False),
        }
        model.named_parameters.return_value = parameters.items()
        return model

    @pytest.fixture
    def phase(self):
        return Mock()

    @pytest.mark.parametrize(
        "config,expected_states",
        [
            # Test case 1: Train only backbone and prototype layers
            (
                {
                    "train_backbone": True,
                    "train_add_on_layers": False,
                    "train_prototype_layer": True,
                    "train_prototype_prediction_head": False,
                    "train_conv_offset": False,
                },
                {
                    "backbone": True,
                    "add_on_layers": False,
                    "prototype_layer": True,
                    "prototype_prediction_head": False,
                    "conv_offset": False,
                },
            ),
            # Test case 2: Train only add_on_layers and prediction head
            (
                {
                    "train_backbone": False,
                    "train_add_on_layers": True,
                    "train_prototype_layer": False,
                    "train_prototype_prediction_head": True,
                    "train_conv_offset": False,
                },
                {
                    "backbone": False,
                    "add_on_layers": True,
                    "prototype_layer": False,
                    "prototype_prediction_head": True,
                    "conv_offset": False,
                },
            ),
            # Test case 3: Train everything except prototype layer
            (
                {
                    "train_backbone": True,
                    "train_add_on_layers": True,
                    "train_prototype_layer": False,
                    "train_prototype_prediction_head": True,
                    "train_conv_offset": True,
                },
                {
                    "backbone": True,
                    "add_on_layers": True,
                    "prototype_layer": False,
                    "prototype_prediction_head": True,
                    "conv_offset": True,
                },
            ),
        ],
    )
    def test_sets_correct_training_layers(self, model, phase, config, expected_states):
        """Test that layers are set trainable according to different configurations"""
        strategy = TrainLayersUsingProtoPNetNames(**config)
        strategy.set_training_layers(model, phase)

        params = dict(model.named_parameters())

        for layer_name, should_train in expected_states.items():
            # Check weight parameters
            if f"{layer_name}.weight" in params:
                assert (
                    params[f"{layer_name}.weight"].requires_grad is should_train
                ), f"{layer_name}.weight should be {should_train}"

            # Check bias parameters
            if f"{layer_name}.bias" in params:
                assert (
                    params[f"{layer_name}.bias"].requires_grad is should_train
                ), f"{layer_name}.bias should be {should_train}"

            # Special case for prototype_layer which has prototypes instead of bias
            if layer_name == "prototype_layer" and f"{layer_name}.prototypes" in params:
                assert (
                    params[f"{layer_name}.prototypes"].requires_grad is should_train
                ), f"{layer_name}.prototypes should be {should_train}"

    def test_handles_unknown_layer(self, model, phase):
        """Test that an unknown layer name raises AssertionError"""
        parameters = dict(model.named_parameters())
        parameters["unknown_layer.weight"] = Mock(requires_grad=False)
        model.named_parameters.return_value = parameters.items()

        strategy = TrainLayersUsingProtoPNetNames(
            train_backbone=True,
            train_add_on_layers=True,
            train_prototype_layer=True,
            train_prototype_prediction_head=True,
            train_conv_offset=False,
        )

        with pytest.raises(AssertionError):
            strategy.set_training_layers(model, phase)


class TestTrainOptimizerLayers:
    @pytest.fixture
    def model(self):
        model = Mock()
        self.params = {
            "layer1.weight": Mock(requires_grad=False),
            "layer2.weight": Mock(requires_grad=False),
            "layer3.weight": Mock(requires_grad=False),
        }
        model.named_parameters.return_value = self.params.items()
        return model

    @pytest.fixture
    def phase_12(self):
        phase = Mock()
        optimizer = Mock()
        # First param group has certain params
        optimizer.param_groups = [{"params": ["layer1.weight", "layer2.weight"]}]
        phase.optimizer = optimizer
        return phase

    @pytest.fixture
    def phase_23(self):
        phase = Mock()
        optimizer = Mock()
        # First param group has certain params
        optimizer.param_groups = [{"params": ["layer2.weight", "layer3.weight"]}]
        phase.optimizer = optimizer
        return phase

    def test_sets_optimizer_params_trainable(self, model, phase_12, phase_23):
        """Test that parameters in optimizer are set trainable"""
        strategy = TrainOptimizerLayers()
        strategy.set_training_layers(model, phase_12)

        params = dict(model.named_parameters())
        assert params["layer1.weight"].requires_grad is True
        assert params["layer2.weight"].requires_grad is True
        assert params["layer3.weight"].requires_grad is False

        strategy.set_training_layers(model, phase_23)

        params = dict(model.named_parameters())
        assert params["layer1.weight"].requires_grad is False
        assert params["layer2.weight"].requires_grad is True
        assert params["layer3.weight"].requires_grad is True

    def test_handles_none_optimizer_params(self, model, phase_12):
        """Test behavior when optimizer params is None"""
        phase_12.optimizer.param_groups = [{"params": None}]

        strategy = TrainOptimizerLayers()
        strategy.set_training_layers(model, phase_12)

        # When params is None, all parameters should be trainable
        params = dict(model.named_parameters())
        assert all(param.requires_grad for param in params.values())

    def test_nested_parameters(self):
        """Test that nested parameters are handled correctly"""
        model = Mock()
        parameters = {
            "layer1": Mock(requires_grad=False),
            "layer1.sublayer": Mock(requires_grad=False),
            "layer1.sublayer.weight": Mock(requires_grad=False),
            "layer1.sublayer.bias": Mock(requires_grad=False),
            "layer2": Mock(requires_grad=False),
            "layer2.weight": Mock(requires_grad=False),
        }
        model.named_parameters.return_value = parameters.items()

        phase = Mock()
        optimizer = Mock()
        # Only include specific nested parameters in optimizer
        optimizer.param_groups = [
            {
                "params": [
                    "layer1.sublayer.weight",
                    "layer1.sublayer.bias",
                    "layer2.weight",
                ]
            }
        ]
        phase.optimizer = optimizer

        strategy = TrainOptimizerLayers()
        strategy.set_training_layers(model, phase)

        params = dict(model.named_parameters())
        # Parent layers should not be trainable
        assert params["layer1"].requires_grad is False
        assert params["layer1.sublayer"].requires_grad is False
        assert params["layer2"].requires_grad is False

        # Specific nested parameters should be trainable
        assert params["layer1.sublayer.weight"].requires_grad is True
        assert params["layer1.sublayer.bias"].requires_grad is True
        assert params["layer2.weight"].requires_grad is True
