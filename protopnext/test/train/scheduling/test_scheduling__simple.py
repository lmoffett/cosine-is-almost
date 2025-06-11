from unittest.mock import Mock, patch

import torch

from protopnet.train.scheduling.scheduling import (
    ProjectPhase,
    PrunePrototypesPhase,
    RSampleInitPhase,
    _NoGradPhaseMixin,
)
from protopnet.train.scheduling.types import PostPhaseSummary


class MockProtoPNet:
    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def project(self, dataloader):
        pass

    def get_prototype_complexity(self):
        return {"complexity": 0.5}

    def prune_prototypes(self):
        pass

    def rsample_init(self, dataloader):
        pass


class TestNoGradPhaseMixin:
    def test_phase_settings(self):
        mixin = _NoGradPhaseMixin()
        model = MockProtoPNet()

        assert torch.is_grad_enabled(), "Gradients should be enabled by default"

        # Test preserving original training state
        model.training = True
        with mixin.phase_settings(model):
            assert not model.training  # Should be in eval mode
            assert not torch.is_grad_enabled()
        assert model.training  # Should restore to original state

        # Test with original state being False
        model.training = False
        with mixin.phase_settings(model):
            assert not model.training
            assert not torch.is_grad_enabled()
        assert not model.training


class TestProjectPhase:
    def test_run_step(self):
        mock_dataloader = Mock(spec=torch.utils.data.DataLoader)
        mock_model = Mock(spec=MockProtoPNet)
        mock_model.get_prototype_complexity.return_value = {"complexity": 0.5}

        phase = ProjectPhase(dataloader=mock_dataloader)
        result = phase.run_step(mock_model, Mock())

        # Verify interactions
        mock_model.project.assert_called_once_with(mock_dataloader)
        mock_model.get_prototype_complexity.assert_called_once()
        assert result == {"complexity": 0.5}

    def test_name_property(self):
        phase = ProjectPhase(dataloader=Mock())
        assert phase.name == "project"

    def test_post_project_metrics(self):
        phase = ProjectPhase(dataloader=Mock())

        mock_metric_logger = Mock()

        phase.after_training(
            Mock(),
            mock_metric_logger,
            Mock(),
            post_phase_summary=PostPhaseSummary(
                name="project",
                first_step=0,
                initial_target_metric=("accuracy", 0.4),
                expected_last_step=10,
                last_step=11,
                final_target_metric=("accuracy", 0.5),
            ),
        )

        assert mock_metric_logger.log_metric.called_once_with(
            "project",
            step=12,
            prototypes_embedded_state=True,
            precalculated_metrics={"accuracy": 0.5},
        )


class TestPrunePrototypesPhase:
    def test_run_step(self):
        mock_model = Mock(spec=MockProtoPNet)
        mock_model.get_prototype_complexity.return_value = {"complexity": 0.3}

        phase = PrunePrototypesPhase()
        result = phase.run_step(mock_model, Mock())

        # Verify interactions
        mock_model.prune_prototypes.assert_called_once()
        mock_model.get_prototype_complexity.assert_called_once()
        assert result == {"complexity": 0.3}

    def test_name_property(self):
        phase = PrunePrototypesPhase()
        assert phase.name == "prune_prototypes"


class TestRSampleInitPhase:
    def test_run_step(self):
        mock_dataloader = Mock(spec=torch.utils.data.DataLoader)
        mock_model = Mock(spec=MockProtoPNet)

        phase = RSampleInitPhase(dataloader=mock_dataloader)
        result = phase.run_step(mock_model, Mock())

        # Verify interactions
        mock_model.rsample_init.assert_called_once_with(mock_dataloader)
        assert result is None

    def test_name_property(self):
        phase = RSampleInitPhase(dataloader=Mock())
        assert phase.name == "rsample_init"


# Integration test showing how the phases work together with the mixin
def test_phase_integration():
    model = MockProtoPNet()
    mock_dataloader = Mock(spec=torch.utils.data.DataLoader)

    # Test all phases in sequence
    phases = [
        ProjectPhase(dataloader=mock_dataloader),
        PrunePrototypesPhase(),
        RSampleInitPhase(dataloader=mock_dataloader),
    ]

    for phase in phases:
        with phase.phase_settings(model):
            assert not model.training  # Should be in eval mode
            assert not torch.is_grad_enabled()  # Should have gradients disabled
            phase.run_step(model, Mock())
        assert model.training  # Should restore to original state
