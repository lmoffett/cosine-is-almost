from protopnet.datasets import dogs


def test_train_dataloaders():
    splits_dataloaders = dogs.train_dataloaders(
        data_path="test/dummy_test_files/test_dataset", part_labels=False
    )

    assert splits_dataloaders.train_loader.batch_size == 95
    assert splits_dataloaders.val_loader.batch_size == 100
    assert splits_dataloaders.project_loader.batch_size == 75
    assert splits_dataloaders.test_loader.batch_size == 100
    assert splits_dataloaders.num_classes == 120

    assert len(splits_dataloaders.train_loader.dataset) == 1
    assert len(splits_dataloaders.val_loader.dataset) == 1
    assert len(splits_dataloaders.project_loader.dataset) == 1
    assert len(splits_dataloaders.test_loader.dataset) == 2
