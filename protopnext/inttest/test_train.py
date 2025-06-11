from .conftest import assert_epoch_training, python


def test_train_protopnet_with_cosine_activation(cifar10_squeezenet1_0_path):
    # NOOP since the fixture itself runs this test
    pass


def test_train_protopnet_with_l2_activation():
    stdout, _ = python(
        "-u -m protopnet train-protopnet --verify --dataset=cifar10 --backbone=squeezenet1_0 --activation-function=l2",
        {"WANDB_MODE": "dryrun"},
    )
    # 3 epochs because it will not pass the random threshold
    assert_epoch_training(stdout, ["warm", "joint", "project", "last_only", "prune"])


def test_train_deformable():
    stdout, _ = python(
        "-u -m protopnet train-deformable --verify --dataset=cifar10 --backbone=squeezenet1_0",
        {"WANDB_MODE": "dryrun"},
    )
    assert_epoch_training(
        stdout, ["warm", "warm_pre_offset", "joint", "project", "last_only"]
    )


def test_train_tesnet():
    stdout, _ = python(
        "-u -m protopnet train-tesnet --verify --dataset=cifar10 --backbone=squeezenet1_0",
        {"WANDB_MODE": "dryrun"},
    )
    assert_epoch_training(stdout, ["warm", "joint", "project", "last_only"])


def test_train_prototree():
    stdout, _ = python(
        "-u -m protopnet train-prototree --verify --dataset=cifar10 --backbone=squeezenet1_0",
        {"WANDB_MODE": "dryrun"},
    )
    assert_epoch_training(stdout, ["warm", "joint", "prune", "project"])


def test_fine_annotation():
    stdout, _ = python(
        "-u -m protopnet train-protopnet --verify --dataset=cifar10_fa --fa-func=l2_norm --fa-coef=0.005",
        {"WANDB_MODE": "dryrun", "PPNXT_DATALOADERS_MODULES": "inttest.datasets"},
    )
    assert_epoch_training(stdout, ["warm", "joint", "project", "last_only", "prune"])


def test_train_stprotopnet():
    stdout, _ = python(
        "-u -m protopnet train-support-trivial --verify --dataset=cifar10 --backbone=squeezenet1_0",
        {"WANDB_MODE": "dryrun"},
    )
    assert_epoch_training(stdout, ["warm", "joint", "project", "last_only"])
