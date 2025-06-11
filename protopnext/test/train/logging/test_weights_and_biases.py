from unittest.mock import patch

import pytest
from torch import tensor

from protopnet.train.logging.weights_and_biases import WeightsAndBiasesTrainLogger


class TestWandbTrainLogger:
    @pytest.fixture
    def wandb_logger(self):
        # Setup the WeightsAndBiasesTrainLogger instance
        return WeightsAndBiasesTrainLogger(
            calculate_best_for=["accu", "sparsity_score"],
        )

    @pytest.fixture
    def mock_wandb_log(self):
        with patch("wandb.log") as mock_wandb_log:
            yield mock_wandb_log

    @pytest.fixture
    def mock_wandb_run(self):
        with patch("wandb.run") as mock_wandb_run:
            yield mock_wandb_run

    @pytest.fixture
    def mock_wandb_plot(self):
        with patch("wandb.plot_table") as mock_wandb_plot:
            yield mock_wandb_plot

    @pytest.fixture
    def epoch_metrics(self):
        return {
            "time": 120,
            "accu": 80.0,
            "n_batches": 10,
            "cross_entropy": 5.0,
            "n_correct": 80,
            "n_examples": 100,
            "is_train": True,
        }

    def test_log_metrics_prototypes_embedded(
        self, wandb_logger, mock_wandb_log, mock_wandb_run
    ):
        # Metrics dict to pass to the log_metrics method
        for key in wandb_logger.val_metrics:
            wandb_logger.val_metrics[key].update(0.0)

        wandb_logger.val_metrics["accu"].update(100.0)  # Simulating an accuracy of 50%

        # Call log_metrics with prototype embedding
        wandb_logger.log_metrics(
            is_train=False,
            prototypes_embedded_state=True,
            precalculated_metrics=None,
            step=2,
        )

        # Assertions to ensure wandb.log was called correctly
        mock_wandb_log.assert_called_once_with(
            {
                "eval/n_examples": tensor(0.0),
                "eval/n_correct": tensor(0.0),
                "eval/n_batches": tensor(0.0),
                "eval/cross_entropy": tensor(0.0),
                "eval/cluster": tensor(0.0),
                "eval/separation": tensor(0.0),
                "eval/fine_annotation": tensor(0.0),
                "eval/accu": tensor(50.0),
                "eval/l1": tensor(0.0),
                "eval/total_loss": tensor(0.0),
                "eval/nll_loss": tensor(0.0),
                "eval/orthogonality_loss": tensor(0.0),
                "eval/contrastive_masked": tensor(0.0),
                "eval/contrastive_unmasked": tensor(0.0),
                "eval/n_unique_proto_parts": tensor(0.0),
                "eval/n_unique_protos": tensor(0.0),
                "eval/grassmannian_orthogonality_loss": tensor(0.0),
            },
            step=2,
            commit=True,
        )

    def test_initialization(self, wandb_logger):
        # Check initial states and defaults
        assert isinstance(wandb_logger.train_metrics, dict)
        assert isinstance(wandb_logger.val_metrics, dict)
        assert wandb_logger.use_ortho_loss is False
        assert wandb_logger.bests["eval"]["accu"]["any"] == float("-inf")
        assert wandb_logger.bests["train"]["accu"]["prototypes_embedded"] == float(
            "-inf"
        )

    def test_end_epoch_precalculated_metrics(
        self,
        wandb_logger,
        epoch_metrics,
        mock_wandb_log,
        mock_wandb_run,
        mock_wandb_plot,
    ):
        wandb_logger.end_epoch(
            epoch_metrics,
            is_train=True,
            epoch_index=0,
            prototype_embedded_epoch=False,
        )

        assert wandb_logger.bests["train"]["accu"]["any"] == 80.0
        assert wandb_logger.bests["train"]["accu"]["prototypes_embedded"] == float(
            "-inf"
        )

        # Check if best prototypes embedded accuracy is updated
        mock_wandb_run.summary.__setitem__.assert_any_call("best/train/accu", 80.0)
        mock_wandb_run.summary.__setitem__.assert_any_call("best/train/accu_step", 0)

        precomputed = {"sparsity_score": 0.5}
        wandb_logger.end_epoch(
            epoch_metrics,
            is_train=True,
            epoch_index=1,
            prototype_embedded_epoch=True,
            precalculated_metrics=precomputed,
        )
        assert wandb_logger.bests["train"]["sparsity_score"]["any"] == 0.5
        assert (
            wandb_logger.bests["train"]["sparsity_score"]["prototypes_embedded"] == 0.5
        )

        # Check if best prototypes embedded accuracy is updated
        mock_wandb_run.summary.__setitem__.assert_any_call(
            "best/train/sparsity_score", 0.5
        )
        mock_wandb_run.summary.__setitem__.assert_any_call(
            "best/train/sparsity_score_step", 1
        )

    def test_update_metrics(self, wandb_logger, epoch_metrics):
        wandb_logger.update_metrics(epoch_metrics, is_train=True)
        assert wandb_logger.train_metrics["n_examples"].compute() == 100
        assert wandb_logger.train_metrics["accu"].compute() == 80.0

    def test_update_bests(self, wandb_logger, mock_wandb_log, mock_wandb_run):
        # Test updating best records
        metrics_dict = {"accu": 0.75}
        wandb_logger.update_bests(
            metrics_dict, step=0, is_train=False, prototype_embedded_epoch=False
        )
        assert wandb_logger.bests["eval"]["accu"]["any"] == 0.75
        assert wandb_logger.bests["eval"]["accu"]["prototypes_embedded"] == float(
            "-inf"
        )

        # Test if updates correctly for prototype-embedded metrics
        wandb_logger.update_bests(
            metrics_dict, step=0, is_train=False, prototype_embedded_epoch=True
        )
        assert wandb_logger.bests["eval"]["accu"]["prototypes_embedded"] == 0.75

    def test_serialize_bests(self, wandb_logger, mock_wandb_log, mock_wandb_run):
        # Ensure serialization of bests works correctly
        wandb_logger.bests["train"]["accu"]["any"] = 80.0
        wandb_logger.bests["train"]["accu"]["prototypes_embedded"] = 75.0
        expected = {
            "best/train/accu": 80.0,
            "best/train/sparsity_score": float("-inf"),
            "best[prototypes_embedded]/train/accu": 75.0,
            "best[prototypes_embedded]/train/sparsity_score": float("-inf"),
            "best/eval/accu": float("-inf"),
            "best/eval/sparsity_score": float("-inf"),
            "best[prototypes_embedded]/eval/accu": float("-inf"),
            "best[prototypes_embedded]/eval/sparsity_score": float("-inf"),
        }
        assert wandb_logger.serialize_bests() == expected, (
            wandb_logger.serialize_bests(),
            expected,
        )
