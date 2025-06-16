import pytest
from dinotool.data import VideoDir, VideoFile, VideoDataset, Video
from dinotool import data


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
    from torchvision import transforms
    import torch

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
    from torchvision import transforms
    import torch
    from torch.utils.data import DataLoader

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


def test_input_pipeline_video_dir():
    from torch.utils.data import DataLoader

    input = data.input_pipeline("dinov2_vits14_reg", "test/data/nasa_frames", patch_size=16, batch_size=2)
    assert isinstance(input["data"], DataLoader)
    assert input["input_size"] == (480, 256)
    assert input["feature_map_size"] == (30, 16)


def test_input_pipeline_video_file():
    from torch.utils.data import DataLoader

    input = data.input_pipeline("dinov2_vits14_reg", "test/data/nasa.mp4", patch_size=16, batch_size=2)
    assert isinstance(input["data"], DataLoader)
    assert input["input_size"] == (480, 256)
    assert input["feature_map_size"] == (30, 16)


def test_input_pipeline_image_file():
    from torch.utils.data import DataLoader
    import torch

    input = data.input_pipeline("dinov2_vits14_reg", "test/data/magpie.jpg", patch_size=16, batch_size=2)
    assert isinstance(input["data"], torch.Tensor)
    assert input["input_size"] == (496, 368)
    assert input["feature_map_size"] == (31, 23)
