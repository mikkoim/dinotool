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


def test_batch_handler():
    from dinotool.model import load_dino_model, PCAModule, DinoFeatureExtractor
    from dinotool.data import input_pipeline
    from dinotool.utils import BatchHandler

    model = load_dino_model("dinov2_vits14_reg")

    input = input_pipeline(
        "test/data/nasa.mp4", patch_size=model.patch_size, batch_size=1
    )
    batch = next(iter(input["data"]))

    extractor = DinoFeatureExtractor(model, input_size=input["input_size"])
    features = extractor(batch["img"])

    pca = PCAModule(n_components=3, feature_map_size=input["feature_map_size"])
    pca.fit(features)

    batch_handler = BatchHandler(input["source"], extractor, pca)

    batch_frames = batch_handler(batch)
    assert batch_frames[0].img.size == (480, 270)
    assert batch_frames[0].features.shape == torch.Size([19, 34, 384])
    assert batch_frames[0].pca.shape == (19, 34, 3)


def test_feature_saving():
    from dinotool.model import load_dino_model, PCAModule, DinoFeatureExtractor
    from dinotool.data import input_pipeline, create_xarray_from_batch_frames
    from dinotool.utils import BatchHandler

    model = load_dino_model("dinov2_vits14_reg")

    input = input_pipeline(
        "test/data/nasa.mp4", patch_size=model.patch_size, batch_size=2
    )
    batch = next(iter(input["data"]))

    extractor = DinoFeatureExtractor(model, input_size=input["input_size"])
    features = extractor(batch["img"])
    pca = PCAModule(n_components=3, feature_map_size=input["feature_map_size"])
    pca.fit(features)

    batch_handler = BatchHandler(input["source"], extractor, pca)
    i = 0
    for batch in input["data"]:
        batch_frames = batch_handler(batch)
        f_data = create_xarray_from_batch_frames(batch_frames)
        i += 1
        if i == 2:
            break
    assert f_data.shape == (2, 19, 34, 384)
    assert all(f_data.frame_idx.values == [2, 3])
