import pytest
import torch

from protopnet.pretrained.vgg_features import VGG_features, cfg


@pytest.mark.parametrize(
    "final_maxpool,embedding_spatial_dim", [(True, 7), (False, 14)]
)
def test_embedding_dim(final_maxpool, embedding_spatial_dim):
    vgg_features = VGG_features(cfg=cfg["D"], final_maxpool=final_maxpool)

    output = vgg_features.forward(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 512, embedding_spatial_dim, embedding_spatial_dim)
