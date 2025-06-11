import pytest

from protopnet.train.logging.tensorboard_logger import TensorBoardLogger


class TestTensorboardTrainLogger:
    @pytest.fixture
    def logger(self):
        return TensorBoardLogger(
            calculate_best_for=["accu", "sparsity_score"],
        )

    @pytest.fixture
    def epoch_metrics(self):
        return {
            "time": 120,
            "accu": 80.0,
            "n_batches": 10,
            "n_examples": 100,
            "cross_entropy": 5.0,
            "n_correct": 80,
            "is_train": True,
        }

    def test_initialization(self, logger):
        # Check initial states and defaults
        assert isinstance(logger.train_metrics, dict)
        assert isinstance(logger.val_metrics, dict)
        assert logger.use_ortho_loss is False
        assert logger.class_specific is True
        assert "train" in logger.bests
        assert "accu" in logger.bests["train"]
        assert logger.bests["train"]["accu"]["any"] == float("-inf")

    def test_end_epoch_division_by_n_batches(self, logger, epoch_metrics):
        logger.end_epoch(
            epoch_metrics,
            is_train=True,
            epoch_index=1,
            prototype_embedded_epoch=False,
        )
        assert (
            epoch_metrics["cross_entropy"] == 0.5
        ), "cross_entropy should be divided by n_batches"

    def test_end_epoch_precalculated_metrics(self, logger, epoch_metrics):
        logger.end_epoch(
            epoch_metrics,
            is_train=True,
            epoch_index=1,
            prototype_embedded_epoch=False,
        )

        assert logger.bests["train"]["accu"]["any"] == 80.0
        assert logger.bests["train"]["accu"]["prototypes_embedded"] == float("-inf")

        precomputed = {"sparsity_score": 0.5}
        logger.end_epoch(
            epoch_metrics,
            is_train=True,
            epoch_index=1,
            prototype_embedded_epoch=True,
            precalculated_metrics=precomputed,
        )
        assert logger.bests["train"]["sparsity_score"]["any"] == 0.5
        assert logger.bests["train"]["sparsity_score"]["prototypes_embedded"] == 0.5

    def test_update_metrics(self, logger, epoch_metrics):
        logger.update_metrics(epoch_metrics, is_train=True)
        assert logger.train_metrics["n_examples"].compute() == 100
        assert logger.train_metrics["accu"].compute() == 80.0

    def test_update_bests(self, logger):
        # Test updating best records
        metrics_dict = {"accu": 0.75}
        logger.update_bests(
            metrics_dict, is_train=False, step=0, prototype_embedded_epoch=False
        )
        assert logger.bests["eval"]["accu"]["any"] == 0.75
        assert logger.bests["eval"]["accu"]["prototypes_embedded"] == float("-inf")

        # Test if updates correctly for prototype-embedded metrics
        logger.update_bests(
            metrics_dict, is_train=True, step=1, prototype_embedded_epoch=True
        )
        assert logger.bests["train"]["accu"]["prototypes_embedded"] == 0.75

    def test_serialize_bests(self, logger):
        # Ensure serialization of bests works correctly
        logger.bests["eval"]["accu"]["any"] = 80.0
        logger.bests["eval"]["accu"]["prototypes_embedded"] = 75.0
        expected = {
            "best/eval/accu": 80.0,
            "best/eval/sparsity_score": float("-inf"),
            "best[prototypes_embedded]/eval/accu": 75.0,
            "best[prototypes_embedded]/eval/sparsity_score": float("-inf"),
            "best/train/accu": float("-inf"),
            "best/train/sparsity_score": float("-inf"),
            "best[prototypes_embedded]/train/accu": float("-inf"),
            "best[prototypes_embedded]/train/sparsity_score": float("-inf"),
        }

        assert logger.serialize_bests() == expected, (
            logger.serialize_bests(),
            expected,
        )
