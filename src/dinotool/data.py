import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Tuple, Dict, List
import cv2
from dataclasses import dataclass
import numpy as np
from torchvision import transforms
import xarray as xr


@dataclass
class FrameData:
    img: Image.Image
    features: torch.Tensor
    pca: np.ndarray
    frame_idx: int
    flattened: bool

    def __post_init__(self):
        if self.flattened:
            if self.features.ndim != 2:
                raise ValueError(f"Expected 2D features, got {self.features.ndim}D")
            if self.pca.ndim != 2:
                raise ValueError(f"Expected 2D PCA, got {self.pca.ndim}D")
        else:
            if self.features.ndim != 3:
                raise ValueError(f"Expected 3D features, got {self.features.shape}")
            if self.pca.ndim != 3:
                raise ValueError(f"Expected 3D PCA, got {self.pca.shape}")


class VideoDir:
    """
    A class to load video frames from a directory.
    The frames are expected to be named in a way that allows them to be sorted
    in the order they were captured (e.g., 01.jpg, 02.jpg, ...).
    """

    def __init__(self, path: str):
        """
        Args:
            path (str): Directory containing video frames.
        """
        self.path = path
        frame_names = [
            p for p in os.listdir(path) if os.path.splitext(p)[-1] in [".jpg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self.frame_names = frame_names

    @property
    def resolution(self):
        """Returns the resolution of the first frame."""
        img = self[0]
        return img.size

    def __repr__(self):
        return f"VideoDir(path={self.path}, frame_count={len(self.frame_names)})"

    def __len__(self):
        """Returns the number of frames in the video."""
        return len(self.frame_names)

    def __getitem__(self, idx):
        frame_name = self.frame_names[idx]
        img = Image.open(f"{self.path}/{frame_name}")
        return img


class VideoFile:
    """
    A class to load video frames from a video file.
    """

    def __init__(self, path: str):
        """
        Args:
            video_file (str): Path to the video file.
        """
        self.path = path
        self.video_capture = cv2.VideoCapture(path)
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def resolution(self):
        """Returns the resolution of the first frame."""
        img = self[0]
        return img.size

    def __repr__(self):
        return f"VideoFile(path={self.path}, frame_count={self.frame_count})"

    def __len__(self):
        """Returns the number of frames in the video."""
        return self.frame_count

    def __getitem__(self, idx):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.video_capture.read()
        if not ret:
            raise IndexError("Frame index out of range")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def __del__(self):
        """Releases the video capture object."""
        self.video_capture.release()
        cv2.destroyAllWindows()


class Video:
    """
    A class to load video frames from a video file or a directory.
    """

    def __init__(self, video_path: str):
        """
        Args:
            video_path (str): Path to the video file or directory containing frames.
        """
        if os.path.isdir(video_path):
            self.video = VideoDir(video_path)
        else:
            self.video = VideoFile(video_path)

    @property
    def resolution(self):
        """Returns the resolution of the first frame."""
        img = self[0]
        return img.size

    @property
    def framerate(self):
        if isinstance(self.video, VideoDir):
            raise ValueError("VideoDir objects have unknown framerate")
        return self.video.video_capture.get(cv2.CAP_PROP_FPS)

    def __repr__(self):
        return f"Video(path={self.video.path}, frame_count={self.video.frame_count})"

    def __len__(self):
        """Returns the number of frames in the video."""
        return len(self.video)

    def __getitem__(self, idx):
        return self.video[idx]


def calculate_dino_dimensions(
    size: Tuple[int, int], patch_size: int = 16
) -> Dict[str, int]:
    """
    Calculates the input dimensions for a image passed to a DINO model, as well as the
    dimensions of the feature map.

    Args:
        size (Tuple[int, int]): The input size (width, height).
        patch_size (int): The size of each patch.

    Returns:
        Dict[str, int]: A dictionary containing the input image width and height,
                        widht and height of the feature map,
                        and the patch size.
    """
    w, h = size[0] - size[0] % patch_size, size[1] - size[1] % patch_size
    return {
        "w": w,
        "h": h,
        "w_featmap": w // patch_size,
        "h_featmap": h // patch_size,
        "patch_size": patch_size,
    }


def input_pipeline(input, patch_size, batch_size=1, resize_size=None):
    """Handles transformation and dimension calculations for video or image input"""
    try:
        img = Image.open(input)
        original_input_size = img.size
        is_image = True
    except (Image.UnidentifiedImageError, IsADirectoryError):
        video = Video(input)
        original_input_size = video.resolution
        is_image = False

    if resize_size is not None:
        original_input_size = resize_size
        print(f"Resizing input to {resize_size}")

    dims = calculate_dino_dimensions(original_input_size, patch_size=patch_size)
    input_size = (dims["w"], dims["h"])
    feature_map_size = (dims["w_featmap"], dims["h_featmap"])

    transform = transforms.Compose(
        [
            transforms.Resize((input_size[1], input_size[0])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if is_image:
        img_tensor = transform(img).unsqueeze(0)
        return {
            "source": img,
            "data": img_tensor,
            "input_size": input_size,
            "feature_map_size": feature_map_size,
        }
    # else
    ds = VideoDataset(video, transform=transform)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return {
        "source": video,
        "data": dataloader,
        "input_size": input_size,
        "feature_map_size": feature_map_size,
    }


class VideoDataset(Dataset):
    def __init__(self, video: Video, transform: nn.Module = None):
        """
        pytorch dataset for video frames.
        Args:
            video (Video): Video object containing frames.
            transform (nn.Module): Transform to apply to each frame.
            size (Tuple[int, int]): The input size (width, height).
            patch_size (int): The size of each patch.
        """
        self.video = video
        if transform is None:
            self.transform = nn.Identity()
        else:
            self.transform = transform

    def __getitem__(self, idx):
        frame = self.video[idx]
        img = self.transform(frame)
        return {"img": img, "frame_idx": idx}

    def __len__(self):
        return len(self.video)


def create_xarray_from_batch_frames(batch_frames: List[FrameData]) -> xr.DataArray:
    tensor = torch.stack([x.features for x in batch_frames])
    frame_idx = [x.frame_idx for x in batch_frames]
    # Assuming the tensor has shape (height, width, feature)
    batch, height, width, feature = tensor.shape

    coords = {
        "frame_idx": frame_idx,
        "y": np.arange(height),
        "x": np.arange(width),
        "feature": np.arange(feature),
    }
    data = xr.DataArray(
        tensor.cpu().numpy(),
        dims=("frame_idx", "y", "x", "feature"),
        coords=coords,
    )
    return data
