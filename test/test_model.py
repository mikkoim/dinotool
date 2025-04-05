import pytest
from dinotool.data import Video, VideoDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .utils import setup_model_and_batch


def test_feature_extractor_basic():
    from dinotool.model import DinoFeatureExtractor

    d = setup_model_and_batch()
    model = d["model"]
    batch = d["batch"]

    extractor = DinoFeatureExtractor(model, input_size=(224, 224))

    # default, flattened, normalized
    out = extractor(batch["img"])
    assert out.shape == torch.Size([1, 256, 384])
    assert torch.allclose(out[0, 0, :].norm().cpu(), torch.tensor([1.0]), atol=1e-5)

    # not flattened, normalized
    out2 = extractor(batch["img"], flattened=False)
    assert isinstance(out, torch.Tensor)
    assert out2.shape == torch.Size([1, 16, 16, 384])
    assert torch.allclose(out2[0, 0, 0, :].norm().cpu(), torch.tensor([1.0]), atol=1e-5)

    # flattened, not normalized
    out3 = extractor(batch["img"], flattened=True, normalized=False)
    assert out3.shape == torch.Size([1, 256, 384])
    assert not torch.allclose(
        out3[0, 0, :].norm().cpu(), torch.tensor([1.0]), atol=1e-5
    )


def test_feature_extractor_flattened():
    from dinotool.model import DinoFeatureExtractor

    d = setup_model_and_batch()
    model = d["model"]
    batch = d["batch"]
    # input size not set
    extractor = DinoFeatureExtractor(model)
    with pytest.raises(ValueError):
        _ = extractor(batch["img"], flattened=False)


def test_pca_module():
    from dinotool.model import PCAModule
    from dinotool.model import DinoFeatureExtractor

    d = setup_model_and_batch()
    model = d["model"]
    batch = d["batch"]

    extractor = DinoFeatureExtractor(model, input_size=(224, 224))
    features = extractor(batch["img"])

    pca = PCAModule(n_components=3)
    pca.fit(features)
    assert pca.pca.mean_.shape == (384,)
    assert pca.pca.components_.shape == (3, 384)
    pca_features = pca.transform(features)
    assert pca_features.shape == (1, 256, 3)


def test_pca_module_nonflat():
    from dinotool.model import PCAModule
    from dinotool.model import DinoFeatureExtractor

    d = setup_model_and_batch()
    model = d["model"]
    batch = d["batch"]

    extractor = DinoFeatureExtractor(model, input_size=(224, 224))
    features = extractor(batch["img"])

    pca = PCAModule(n_components=3, feature_map_size=(16, 16))
    pca.fit(features)
    pca_features = pca.transform(features, flattened=False)
    assert pca_features.shape == (1, 16, 16, 3)
