"""End-to-end tests for the DinoToolModel Python API."""

from dinotool import DinoToolModel
from dinotool.data import LocalFeatures
from PIL import Image
import torch
import numpy as np
import pytest


@pytest.fixture(scope="module")
def model():
    return DinoToolModel("vit-s", verbose=False)


def test_api_full_features(model):
    img = Image.open("test/data/bird1.jpg")
    transform = model.get_transform(img.size)
    img_tensor = transform.transform(img).unsqueeze(0)

    features = model(img_tensor, features="full")
    assert isinstance(features, LocalFeatures)
    assert len(features.shape) == 4  # (b, h, w, f)
    assert features.shape[0] == 1
    assert features.shape[3] == 384


def test_api_flat_features(model):
    img = Image.open("test/data/bird1.jpg")
    transform = model.get_transform(img.size)
    img_tensor = transform.transform(img).unsqueeze(0)

    features = model(img_tensor, features="flat")
    assert isinstance(features, LocalFeatures)
    assert len(features.shape) == 3  # (b, h*w, f)
    assert features.shape[0] == 1
    assert features.shape[2] == 384


def test_api_frame_features(model):
    img = Image.open("test/data/bird1.jpg")
    transform = model.get_transform(img.size)
    img_tensor = transform.transform(img).unsqueeze(0)

    features = model(img_tensor, features="frame")
    assert features.shape == torch.Size([1, 384])
    assert torch.allclose(features[0].norm().cpu(), torch.tensor(1.0), atol=1e-5)


def test_api_unnormalized(model):
    img = Image.open("test/data/bird1.jpg")
    transform = model.get_transform(img.size)
    img_tensor = transform.transform(img).unsqueeze(0)

    features_norm = model(img_tensor, features="flat", normalized=True)
    features_raw = model(img_tensor, features="flat", normalized=False)

    # Normalized features should have unit norm per patch
    norms = features_norm.tensor[0].norm(dim=-1).cpu()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Unnormalized features generally won't have unit norm
    raw_norms = features_raw.tensor[0].norm(dim=-1).cpu()
    assert not torch.allclose(raw_norms, torch.ones_like(raw_norms), atol=1e-5)


def test_api_pca(model):
    img = Image.open("test/data/bird1.jpg")
    transform = model.get_transform(img.size)
    img_tensor = transform.transform(img).unsqueeze(0)

    features = model(img_tensor, features="full")
    pca_result = model.pca(features, n_components=3)

    assert isinstance(pca_result, np.ndarray)
    assert pca_result.shape[2] == 3  # 3 PCA components
    # PCA output is min-max normalized to [0, 1]
    assert np.allclose(pca_result.min(), 0.0)
    assert np.allclose(pca_result.max(), 1.0)


def test_api_batch_processing(model):
    img1 = Image.open("test/data/bird1.jpg")
    img2 = Image.open("test/data/bird2.jpg")

    # Both images need the same transform size for batching
    transform = model.get_transform((224, 224))
    batch = torch.stack([
        transform.transform(img1),
        transform.transform(img2),
    ])

    features = model(batch, features="full")
    assert features.shape[0] == 2
    assert features.shape[3] == 384

    # Each patch vector should be normalized
    norms = features.tensor[0, 0, :, :].norm(dim=-1).cpu()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_api_pca_batch(model):
    transform = model.get_transform((224, 224))
    batch = torch.stack([
        transform.transform(Image.open("test/data/bird1.jpg")),
        transform.transform(Image.open("test/data/bird2.jpg")),
    ])

    features = model(batch, features="full")
    pca_result = model.pca(features, n_components=3)

    assert pca_result.shape[0] == 2
    assert pca_result.shape[3] == 3


def test_api_available_models():
    models = DinoToolModel.available_models()
    assert isinstance(models, dict)
    assert "vit-s" in models
    assert "siglip2" in models
    assert "clip" in models
    assert "dinov3-s" in models
    assert "radio-b" in models


def test_api_repr(model):
    r = repr(model)
    assert "DinoToolModel" in r
    assert "dinov2_vits14_reg" in r
