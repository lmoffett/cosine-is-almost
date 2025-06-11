import copy
import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from protopnet.activations import CosPrototypeActivation, L2Activation
from protopnet.backbones import construct_backbone
from protopnet.embedding import AddonLayers
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
from protopnet.prediction_heads import LinearClassPrototypePredictionHead
from protopnet.prototype_layers import (
    ClassAwarePrototypeLayer,
    PrototypeFromSampleMeta,
    PrototypeWithMeta,
)
from protopnet.prototypical_part_model import ProtoPNet, param_hash


class ShortCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, use_ind_as_label=False, *args, **kwargs):
        super(ShortCIFAR10, self).__init__(*args, **kwargs)
        self.use_ind_as_label = use_ind_as_label

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: {'img': image, 'target': target) where target is index of the target class.
        """
        if self.use_ind_as_label:
            # This is useful for class specific push, where we
            # want to make sure we have at least one image from
            # each class.
            img, target = self.data[index], torch.tensor(index)
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"img": img, "target": target, "sample_id": index}


@pytest.fixture
def mock_model():
    # Create a mock for the model
    mock_model = MagicMock()

    # Mock the prototype info dict (sample_id and other info can be modified as needed)
    mock_model.prototype_layer.prototype_meta = [
        PrototypeFromSampleMeta(
            sample_id=123, target=1, hash="12", latent_patches=torch.tensor([1, 0])
        ),
        PrototypeFromSampleMeta(
            sample_id=456, target=2, hash="45", latent_patches=torch.tensor([0, 1])
        ),
    ]

    # Mock the weight of the class connection layer
    mock_weights = torch.tensor([[0.9, 0.1], [0.6, 0.4]])

    # Mock the class connection layer in the prototype prediction head
    mock_model.prototype_prediction_head.class_connection_layer.weight = mock_weights

    # Return the mocked model object
    return mock_model


def test_cos_model_construction():
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = ClassAwarePrototypeLayer(
        activation_function=cosine_activation,
        prototype_class_identity=prototype_class_identity,
    )

    prediction_head = LinearClassPrototypePredictionHead(
        prototype_class_identity=prototype_class_identity
    )
    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        cosine_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )

    input = torch.randn(10, 3, 224, 224)
    logits = protopnet.forward(input)["logits"]

    assert logits.shape == (10, 3)


def test_L2_model_construction():
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    # Test construction with L2 activation ----------------
    for num_addon_layers in [3, 1, 0, 2]:
        add_on_layers = AddonLayers(
            num_prototypes=3 * 2,
            input_channels=512,
            proto_channel_multiplier=2**0,
            num_addon_layers=num_addon_layers,
        )
    l2_activation = L2Activation()

    prototype_layer = ClassAwarePrototypeLayer(
        activation_function=l2_activation,
        prototype_class_identity=prototype_class_identity,
    )

    prediction_head = LinearClassPrototypePredictionHead(
        prototype_class_identity=prototype_class_identity
    )
    protopnet = ProtoPNet(
        backbone,
        add_on_layers,
        l2_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )

    input = torch.randn(10, 3, 224, 224)
    logits = protopnet.forward(input)["logits"]

    assert logits.shape == (10, 3)


@pytest.fixture
def dataloader(seed, temp_root_dir):
    dataset = ShortCIFAR10(
        root=os.environ.get("CIFAR10_DIR", temp_root_dir / "data"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        generator=torch.Generator().manual_seed(seed),
    )

    return dataloader


@pytest.fixture
def pseudo_label_dataloader(seed, temp_root_dir):
    dataset = ShortCIFAR10(
        root=os.environ.get("CIFAR10_DIR", temp_root_dir / "data"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
        use_ind_as_label=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2,
        generator=torch.Generator().manual_seed(seed),
    )

    return dataloader


def test_class_specific_project(dataloader, pseudo_label_dataloader):
    num_classes = 10
    ppnet1 = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(10 * 2),
        activation=CosPrototypeActivation(),
        num_classes=num_classes,
        num_prototypes_per_class=1,
    )
    ppnet1.to("cpu")
    prototypes = ppnet1.prototype_tensors

    def metadatafy_tensors(tensors):
        return [
            PrototypeWithMeta(
                prototype_tensor=tensors[i],
                meta=PrototypeFromSampleMeta(
                    sample_id=b"test_id",
                    target=i % num_classes,
                    hash=b"hash",
                    latent_patches=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
                ),
            )
            for i in range(tensors.shape[0])
        ]

    # Setting all prototypes to be equal
    ppnet1.prototype_layer.set_prototypes(metadatafy_tensors(prototypes))

    with torch.no_grad():
        project_return = ppnet1.project(pseudo_label_dataloader)

    assert ppnet1.prototypes_embedded() == True
    prototypes = ppnet1.prototype_tensors

    # Since each prototype belongs to a different class, they should be
    # forced to push onto different stuff, even though they started out
    # identical
    for other_proto in range(1, 10):
        assert not torch.allclose(prototypes[0], prototypes[other_proto])

    for i, (returned_meta, saved_meta) in enumerate(
        zip(project_return, ppnet1.prototype_layer.prototype_meta)
    ):
        assert returned_meta.as_meta() == saved_meta
        assert returned_meta.embedding.norm() > 1e-5
        assert returned_meta.img.norm() > 1e-5
        assert len(returned_meta.hash) > 16
        assert returned_meta.sample_id == i
        assert returned_meta.target == i


def test_not_class_specific_project(dataloader, pseudo_label_dataloader):
    num_classes = 10
    ppnet1 = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(10 * 2),
        activation=CosPrototypeActivation(),
        num_classes=num_classes,
        num_prototypes_per_class=1,
    )
    ppnet1.prototype_layer.class_specific_project = False
    ppnet1.to("cpu")
    prototypes = ppnet1.prototype_tensors

    def metadatafy_tensor(tensor, size):
        return [
            PrototypeWithMeta(
                prototype_tensor=tensor,
                meta=PrototypeFromSampleMeta(
                    sample_id=b"test_id",
                    target=i % num_classes,
                    hash=b"hash",
                    latent_patches=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
                ),
            )
            for i in range(size)
        ]

    # Setting all prototypes to be equal
    ppnet1.prototype_layer.set_prototypes(
        metadatafy_tensor(prototypes[0], prototypes.shape[0])
    )

    with torch.no_grad():
        project_return = ppnet1.project(pseudo_label_dataloader)

    assert ppnet1.prototypes_embedded() == True
    prototypes = ppnet1.prototype_tensors

    # Since we don't care about classes, every prototype should project onto the
    # same thing, since they started out identical
    for other_proto in range(1, 10):
        assert torch.allclose(prototypes[0], prototypes[other_proto])

    # For class specific, all the prototypes should project onto the same thing
    closest_sample_id = project_return[0].sample_id
    closest_sample_label = project_return[0].target

    # check that the metadata is updated correctly
    for returned_meta, saved_meta in zip(
        project_return, ppnet1.prototype_layer.prototype_meta
    ):
        assert returned_meta.as_meta() == saved_meta
        assert returned_meta.embedding.norm() > 1e-5
        assert returned_meta.img.norm() > 1e-5
        assert len(returned_meta.hash) > 16
        assert returned_meta.sample_id == closest_sample_id
        assert returned_meta.target == closest_sample_label


@pytest.mark.parametrize("class_specific_project", (True, False))
def test_project_idempotency(
    class_specific_project, dataloader, pseudo_label_dataloader
):
    num_prototypes = 10 * 1
    ppnet1 = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(num_prototypes),
        activation=CosPrototypeActivation(),
        num_classes=10,
        num_prototypes_per_class=1,
    )
    ppnet1.prototype_layer.class_specific_project = class_specific_project
    ppnet1.to("cpu")

    with torch.no_grad():
        ppnet1.project(pseudo_label_dataloader)

    first_project_vectors = ppnet1.prototype_tensors.clone()
    first_project_meta = copy.deepcopy(ppnet1.prototype_layer.prototype_meta)

    with torch.no_grad():
        ppnet1.project(pseudo_label_dataloader)

    push_diff = first_project_vectors - ppnet1.prototype_tensors
    push_diff_norm = torch.norm(push_diff)
    assert push_diff_norm < 1e-5, (push_diff, push_diff_norm)

    for i in range(0, num_prototypes):
        orig_metadata = first_project_meta[i]
        orig_sample_id = orig_metadata.sample_id
        new_metadata = ppnet1.prototype_layer.prototype_meta[i]
        assert new_metadata.sample_id == orig_sample_id
        assert new_metadata.hash == orig_metadata.hash
        assert new_metadata.target == orig_metadata.target


@pytest.mark.parametrize("class_specific_project", (True, False))
def test_project_determinism(class_specific_project, pseudo_label_dataloader):
    # ppnet = model.construct_PPNet(base_architecture="resnet18", device="cpu")
    ppnet = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(10 * 2),
        activation=CosPrototypeActivation(),
        num_classes=10,
        num_prototypes_per_class=2,
    )
    ppnet.prototype_layer.class_specific_project = class_specific_project

    ppnet1 = ppnet
    ppnet1.to("cpu")

    ppnet2 = copy.deepcopy(ppnet)
    ppnet2.to("cpu")

    with torch.no_grad():
        ppnet1.project(pseudo_label_dataloader)

    assert ppnet1.prototypes_embedded() == True
    with torch.no_grad():
        ppnet2.project(pseudo_label_dataloader)

    push_diff = ppnet2.prototype_tensors - ppnet1.prototype_tensors
    push_diff_norm = torch.norm(push_diff)
    assert push_diff_norm < 1e-5, (push_diff, push_diff.sum(), push_diff_norm)
    for i in range(0, ppnet1.prototype_layer.num_prototypes):
        assert (
            ppnet1.prototype_layer.prototype_meta[i]
            == ppnet2.prototype_layer.prototype_meta[i]
        )


@pytest.mark.parametrize("class_specific_project", (True, False))
def test_that_there_is_a_maximal_image_for_each_activation(
    class_specific_project, dataloader, pseudo_label_dataloader
):
    ppnet = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(
            10 * 2, proto_channel_multiplier=0, num_addon_layers=0
        ),
        activation=CosPrototypeActivation(),
        num_classes=10,
        num_prototypes_per_class=2,
    )
    ppnet.prototype_layer.class_specific_project = class_specific_project
    ppnet.to("cpu")
    ppnet.eval()

    def similarity_score_to_each_prototype(ppnet):
        # device = ppnet.module.device
        global_max = torch.zeros(ppnet.prototype_layer.num_prototypes)
        for data_dict in dataloader:
            image = data_dict["img"].to("cpu")
            label = data_dict["target"].to("cpu")
            # batch x num_prototypes x height x width

            activations = ppnet.prototype_layer.activation_function(
                ppnet.backbone(image), ppnet.prototype_tensors
            )

            max_activation = activations.amax(dim=(0, 2, 3))
            global_max = torch.maximum(global_max, max_activation)

        return global_max

    pre_project_activations = similarity_score_to_each_prototype(ppnet)

    # this is a bug in renormalizing the prototype activations
    # once that's fixed, we should remove this
    # FIXME - once we move to the new prototype activation function, this should be removed

    assert not torch.all(pre_project_activations > 0.99), pre_project_activations

    with torch.no_grad():
        ppnet.project(pseudo_label_dataloader)

    post_project_activations = similarity_score_to_each_prototype(ppnet)

    if class_specific_project:
        # since we're using a truncated dataset, classes not in the dataset will have low activations
        assert torch.all(
            torch.logical_or(
                (post_project_activations > 0.99),
                (post_project_activations < 0.0001),
            )
        )

    else:
        assert torch.all(post_project_activations > 0.99), (
            post_project_activations.min(),
            post_project_activations.max(),
            post_project_activations.mean(),
            post_project_activations.std(),
        )


def test_describe_prototypes(mock_model):
    # Set the method to the mock model
    mock_model.describe_prototypes = ProtoPNet.describe_prototypes.__get__(mock_model)

    # Call the method
    result = mock_model.describe_prototypes()

    # Expected output for this test based on mock data
    expected_output = (
        "\nPrototype 0 comes from sample 123."
        "\n\tIt has highest class connection to class 0 with a class connection vector of:"
        "\n\t\ttensor([0.9000, 0.6000])"
        "\nPrototype 1 comes from sample 456."
        "\n\tIt has highest class connection to class 1 with a class connection vector of:"
        "\n\t\ttensor([0.1000, 0.4000])"
    )

    # Assert that the result matches the expected output
    assert result.strip() == expected_output.strip()


def test_identical_layers_same_hash():
    layer1 = nn.Linear(10, 5)
    layer2 = nn.Linear(10, 5)

    # Set identical weights and biases
    with torch.no_grad():
        layer2.weight.copy_(layer1.weight)
        layer2.bias.copy_(layer1.bias)

    hash1 = param_hash(layer1)
    hash2 = param_hash(layer2)
    assert hash1 == hash2


def test_different_layers_different_hash():
    layer1 = nn.Linear(10, 5)
    layer2 = nn.Linear(10, 5)  # Different random initialization

    hash1 = param_hash(layer1)
    hash2 = param_hash(layer2)
    assert hash1 != hash2


def test_small_parameter_changes():
    layer = nn.Linear(10, 5)
    original_hash = param_hash(layer)

    # Make a tiny change that should be rounded away
    with torch.no_grad():
        layer.weight.add_(torch.randn_like(layer.weight) * 1e-10)

    new_hash = param_hash(layer)
    assert original_hash == new_hash  # Should be same due to rounding


def test_significant_parameter_changes():
    layer = nn.Linear(10, 5)
    original_hash = param_hash(layer)

    # Make a significant change
    with torch.no_grad():
        layer.weight.add_(torch.randn_like(layer.weight) * 0.1)

    new_hash = param_hash(layer, precision=8)
    assert original_hash != new_hash


def test_precision_affects_hash():
    layer = nn.Linear(10, 5)

    hash1 = param_hash(layer, precision=3)
    hash2 = param_hash(layer, precision=8)
    assert hash1 != hash2


@pytest.mark.cuda
def test_device_independence():
    layer_cpu = nn.Linear(10, 5)
    layer_gpu = nn.Linear(10, 5).cuda()

    # Set identical weights and biases
    with torch.no_grad():
        layer_gpu.weight.copy_(layer_cpu.weight.cuda())
        layer_gpu.bias.copy_(layer_cpu.bias.cuda())

    hash_cpu = param_hash(layer_cpu)
    hash_gpu = param_hash(layer_gpu)
    assert hash_cpu == hash_gpu, (hash_cpu, hash_gpu)


def test_nested_layers():
    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 2)

    model1 = NestedModule()
    model2 = NestedModule()

    # Set identical parameters
    with torch.no_grad():
        model2.linear1.weight.copy_(model1.linear1.weight)
        model2.linear1.bias.copy_(model1.linear1.bias)
        model2.linear2.weight.copy_(model1.linear2.weight)
        model2.linear2.bias.copy_(model1.linear2.bias)

    hash1 = param_hash(model1)
    hash2 = param_hash(model2)
    assert hash1 == hash2


def test_zero_param_layer():
    class EmptyLayer(nn.Module):
        def __init__(self):
            super().__init__()

    layer = EmptyLayer()
    hash_value = param_hash(layer)
    assert isinstance(hash_value, str)
    assert len(hash_value) > 0  # Should still return a valid hash


def test_different_precisions():
    layer = nn.Linear(10, 5)

    # Test multiple precision levels
    precisions = [1, 3, 5, 8, 10]
    hashes = [param_hash(layer, precision=p) for p in precisions]

    # All hashes should be different
    assert len(set(hashes)) == len(precisions)
