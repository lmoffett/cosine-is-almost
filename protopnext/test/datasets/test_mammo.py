import os
import shutil

import numpy as np

from protopnet.datasets import mammo


def test_train_dataloaders(temp_dir):
    base_dir = temp_dir / "mammo-dataset"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir()

    num_classes = 5
    batch_size = 2

    idx = 0

    for split in ["train", "push", "validation"]:
        split_dir = base_dir / split
        split_dir.mkdir()

        for label in range(num_classes):
            class_dir = split_dir / str(label)
            class_dir.mkdir()

            for _ in range(batch_size):
                data = np.random.rand(
                    np.random.randint(100, 501), np.random.randint(100, 501)
                )
                file_name = f"tmp_file_{idx}.npy"
                file_path = os.path.join(class_dir, file_name)
                np.save(file_path, data)

                idx += 1

    splits_dataloaders = mammo.train_dataloaders(
        data_path=base_dir, with_fa=False, with_aux=False
    )

    assert splits_dataloaders.train_loader.batch_size == 95
    assert splits_dataloaders.val_loader.batch_size == 100
    assert splits_dataloaders.project_loader.batch_size == 75
    assert splits_dataloaders.num_classes == 5

    assert len(splits_dataloaders.train_loader.dataset) == num_classes * batch_size
    assert len(splits_dataloaders.val_loader.dataset) == num_classes * batch_size
    assert len(splits_dataloaders.project_loader.dataset) == num_classes * batch_size
