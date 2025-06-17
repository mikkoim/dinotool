import pytest
from dinotool.data import VideoDir, VideoFile, VideoDataset, Video
from dinotool import data
from torchvision import transforms
import torch
from torch.utils.data import DataLoader


def test_video_dir():
    video = VideoDir("test/data/nasa_frames")
    assert len(video) == 90
    assert repr(video) == "VideoDir(path=test/data/nasa_frames, frame_count=90)"
    assert video[0] is not None
    assert video[0].size == (480, 270)
    assert video.resolution == (480, 270)
    with pytest.raises(IndexError):
        _ = video[1000]


def test_video_file():
    video = VideoFile("test/data/nasa.mp4")
    assert len(video) == 90
    assert repr(video) == "VideoFile(path=test/data/nasa.mp4, frame_count=90)"
    assert video[0] is not None
    assert video[0].size == (480, 270)
    assert video.resolution == (480, 270)
    with pytest.raises(IndexError):
        _ = video[1000]


def test_video():
    video = Video("test/data/nasa.mp4")
    assert len(video) == 90
    assert repr(video) == "Video(path=test/data/nasa.mp4, frame_count=90)"
    assert video[0] is not None
    assert video[0].size == (480, 270)
    assert video.resolution == (480, 270)
    with pytest.raises(IndexError):
        _ = video[1000]


def test_calculate_dino_dimensions():
    size = (640, 480)
    patch_size = 16
    d = data.calculate_dino_dimensions(size, patch_size)
    assert d["w"] == 640
    assert d["h"] == 480
    assert d["w_featmap"] == 40
    assert d["h_featmap"] == 30
    assert d["patch_size"] == 16

    size = (1000, 900)
    patch_size = 16
    d = data.calculate_dino_dimensions(size, patch_size)

    assert d["w"] == 992
    assert d["h"] == 896
    assert d["w_featmap"] == 62
    assert d["h_featmap"] == 56
    assert d["patch_size"] == 16

    size = (1280, 720)
    patch_size = 14
    d = data.calculate_dino_dimensions(size, patch_size)
    assert d["w"] == 1274
    assert d["h"] == 714
    assert d["w_featmap"] == 91
    assert d["h_featmap"] == 51
    assert d["patch_size"] == 14


def test_video_dataset_no_transform():
    video = Video("test/data/nasa.mp4")
    ds = VideoDataset(video)
    assert len(ds) == 90
    assert ds[0]["img"] is not None
    assert ds[0]["img"].size == (480, 270)
    assert ds[0]["frame_idx"] == 0
    assert ds[1]["img"] is not None


def test_video_dataset_simple_transform():

    video = Video("test/data/nasa.mp4")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    ds = VideoDataset(video, transform=transform)
    assert len(ds) == 90
    assert ds[0]["img"] is not None
    assert ds[0]["img"].shape == torch.Size([3, 224, 224])
    assert ds[0]["frame_idx"] == 0


def test_video_dataset_dataloader():

    video = Video("test/data/nasa.mp4")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    ds = VideoDataset(video, transform=transform)
    dataloader = DataLoader(ds, batch_size=8, shuffle=False)
    assert len(dataloader) == 12
    batch = next(iter(dataloader))
    assert batch["img"].shape == torch.Size([8, 3, 224, 224])
    assert batch["frame_idx"].shape == torch.Size([8])
    assert batch["frame_idx"][0] == 0


def test_input_processor_video_dir():

    input_data = data.InputProcessor("dinov2_vits14_reg", "test/data/nasa_frames", patch_size=16, batch_size=2).process()
    assert isinstance(input_data.data, DataLoader)
    assert input_data.input_size == (480, 256)
    assert input_data.feature_map_size == (30, 16)


def test_input_processor_video_file():

    input_data = data.InputProcessor("dinov2_vits14_reg", "test/data/nasa.mp4", patch_size=16, batch_size=2).process()
    assert isinstance(input_data.data, DataLoader)
    assert input_data.input_size == (480, 256)
    assert input_data.feature_map_size == (30, 16)


def test_input_processor_image_file():

    input_data = data.InputProcessor("dinov2_vits14_reg", "test/data/magpie.jpg", patch_size=16, batch_size=2).process()
    assert isinstance(input_data.data, torch.Tensor)
    assert input_data.input_size == (496, 368)
    assert input_data.feature_map_size == (31, 23)

def test_local_features():
    tensor = torch.rand(4, 20, 30, 100)  # Simulated features for 10 frames
    features = data.LocalFeatures(tensor)

    for i in range(4):
        torch.testing.assert_close(features[i].tensor, tensor[i].unsqueeze(0))
    assert features.shape == (4, 20, 30, 100)
    assert features.flat().shape == (4, 600, 100)
    assert features.full().shape == (4, 20, 30, 100)
    assert features.tensor.shape == (4, 20, 30, 100)
    assert features.normalize().tensor[0,0,0,:].norm() == pytest.approx(1.0, rel=1e-5)
    assert features.normalize().tensor[0,3,4,:].norm() == pytest.approx(1.0, rel=1e-5)
