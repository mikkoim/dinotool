import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Union
import cv2
from dataclasses import dataclass
import numpy as np
from torchvision import transforms
import xarray as xr
from einops import rearrange
import pandas as pd
from collections import defaultdict


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
            p for p in os.listdir(path) 
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self.frame_names = frame_names
        self.frame_count = len(frame_names)

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
        frame_path = os.path.join(self.path, frame_name)
        img = Image.open(frame_path).convert("RGB")
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
        if hasattr(self, 'video_capture'):
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
        self.path = video_path
        if os.path.isdir(video_path):
            self.video = VideoDir(video_path)
        else:
            self.video = VideoFile(video_path)

    @property
    def resolution(self):
        """Returns the resolution of the first frame."""
        return self.video.resolution

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

class ImageDirectory:
    """
    A class to load images from a directory.
    The images can be any format supported by PIL and of various sizes.
    """

    def __init__(self, path: str):
        """
        Args:
            path (str): Directory containing images.
        """
        self.path = path
        self.image_names = [
            p for p in os.listdir(path) 
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        ]
        self.image_names.sort()  # Sort images by name
        self.image_count = len(self.image_names)
        self.filename_map = {name: idx for idx, name in enumerate(self.image_names)}

    def __repr__(self):
        return f"ImageDirectory(path={self.path}, image_count={self.image_count})"

    def __len__(self):
        """Returns the number of images in the directory."""
        return self.image_count

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.path, image_name)
        img = Image.open(image_path).convert("RGB")
        return img
    def get_by_name(self, name: str) -> Image.Image:
        """
        Get an image by its name.
        Args:
            name (str): Name of the image file.
        Returns:
            Image.Image: The image object.
        """
        if name not in self.filename_map:
            raise ValueError(f"Image {name} not found in directory {self.path}")
        idx = self.filename_map[name]
        return self.__getitem__(idx)

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
                        width and height of the feature map,
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

@dataclass
class OpenCLIPTransform:
    transform: nn.Module
    resize_size: Optional[Tuple[int, int]] = None
    feature_map_size: Optional[Tuple[int, int]] = None

@dataclass
class DINOTransform:
    transform: nn.Module
    resize_size: Optional[Tuple[int, int]] = None
    feature_map_size: Optional[Tuple[int, int]] = None

class TransformFactory:
    """
    Factory class to create transforms for feature extraction models.
    """
    def __init__(self,
                 model_name,
                 patch_size: int) -> nn.Module:
        """
        Get the appropriate transform for the model based on its name and input size.
        Args:
            model_name (str): Name of the model.
            patch_size (int): Patch size for the model.
        Returns:
            nn.Module: A transform that can be applied to images.
        """
        self.model_name = model_name
        self.patch_size = patch_size

        if self.model_name.startswith("hf-hub:timm"):
            self.model_type = "openclip"
        else:
            self.model_type = "dino"

        self.transform = None
        self._transform_cache = dict()
    
    def __repr__(self):
        return f"TransformFactory(model_name={self.model_name}, patch_size={self.patch_size}, model_type={self.model_type})"

    def get_openclip_transform(self) -> nn.Module:
        if self.transform is not None:
            # If a transform is already set, return it
            return self.transform

        from open_clip import create_model_from_pretrained
        _, transform = create_model_from_pretrained(self.model_name)
        # Pass a dummy image to get the resize size
        dummy_transformed = transform(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        resize_size = (dummy_transformed.shape[2], dummy_transformed.shape[1])

        dims = calculate_dino_dimensions(resize_size, patch_size=self.patch_size)
        model_input_size = (dims["w"], dims["h"])
        feature_map_size = (dims["w_featmap"], dims["h_featmap"])

        self.transform = OpenCLIPTransform(
            transform=transform,
            resize_size=model_input_size,
            feature_map_size=feature_map_size
        )
        return self.transform

    def get_dino_transform(self, input_size: Tuple[int, int]) -> nn.Module:
        if input_size in self._transform_cache:
            # If a transform for this input size is already cached, return it
            return self._transform_cache[input_size]

        dims = calculate_dino_dimensions(input_size, patch_size=self.patch_size)
        model_input_size = (dims["w"], dims["h"])
        feature_map_size = (dims["w_featmap"], dims["h_featmap"])

        transform = transforms.Compose([
            transforms.Resize((model_input_size[1], model_input_size[0])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = DINOTransform(
            transform=transform,
            resize_size=model_input_size,
            feature_map_size=feature_map_size
        )
        self._transform_cache[input_size] = self.transform
        return self.transform

    def get_transform(self, input_size: Tuple[int, int]) -> nn.Module:
        if self.model_type == "openclip":
            return self.get_openclip_transform()
        elif self.model_type == "dino":
            return self.get_dino_transform(input_size)

class InputProcessor:
    """
    Class to handle input processing for feature extraction models.
    This class supports single images, video files, and directories of images.
    Args:
        model_name: Name of the model
        input_path: Path to input (image file, video file, or directory)
        patch_size: Patch size for the model
        batch_size: Batch size for processing
        resize_size: Optional size to resize all images to
    """

    def __init__(self, model_name: str, input_path: str, patch_size: int, batch_size: int = 1, resize_size: Optional[Tuple[int, int]] = None):
        self.model_name = model_name
        self.input_path = input_path
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.resize_size = resize_size

        self.source, self.input_type = self.find_source(input_path)

    @staticmethod
    def find_source(input_path: str):
        """
        Setup processing based on input type
        """
        try:
            source = Image.open(input_path).convert("RGB")
            return source, "single_image"
        except (Image.UnidentifiedImageError, IsADirectoryError):
            if os.path.isdir(input_path):
                try:
                    source = Video(input_path)
                    return source, "video_dir"
                except ValueError:
                    source = ImageDirectory(input_path)
                    return source, "image_directory"
            else:
                source = Video(input_path)
                return source, "video_file"
        except Exception as e:
            raise ValueError(f"Could not identify input type for {input_path}: {e}")

    def process(self):
        if self.input_type == "image_directory":
            return self.process_varying_size()
        else:
            return self.process_fixed_size()
    
    def process_varying_size(self):
        """Varying size processing for image directories, with batch_size=1.
        If the transform is set with a fixed size, batching can still be used.
        """
        transform_factory = TransformFactory(
            model_name=self.model_name,
            patch_size=self.patch_size
        )
        if self.resize_size is not None:
            print(f"Resizing all input to {self.resize_size}")
        ds = ImageDirectoryDataset(self.source, transform_factory=transform_factory, resize_size=self.resize_size)
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        return {
            "source": self.source,
            "data": dataloader,
            "input_size": None,
            "feature_map_size": None,
            "input_type": self.input_type,
        }

    def process_fixed_size(self):
        """
        Process the input based on its type and return the transformed data.
        """
        original_input_size = self._find_original_input_size()
        print(f"Original input size: {original_input_size}")

        if self.resize_size is not None:
            original_input_size = self.resize_size
            print(f"Resizing input to {self.resize_size}")

        transform_factory = TransformFactory(
            model_name=self.model_name,
            patch_size=self.patch_size
        )
        self.transform = transform_factory.get_transform(original_input_size)
        print(f"Model input size: {self.transform.resize_size}")
        print(f"Feature map size: {self.transform.feature_map_size}")

        if self.input_type == "single_image":
            img_tensor = self.transform.transform(self.source).unsqueeze(0)
            return {
                "source": self.source,
                "data": img_tensor,
                "input_size": self.transform.resize_size,
                "feature_map_size": self.transform.feature_map_size,
                "input_type": self.input_type,
            }
        elif self.input_type in ["video_dir", "video_file"]:
            ds = VideoDataset(self.source, transform=self.transform.transform)
            dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            return {
                "source": self.source,
                "data": dataloader,
                "input_size": self.transform.resize_size,
                "feature_map_size": self.transform.feature_map_size,
                "input_type": self.input_type,
            }
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")

    def _find_original_input_size(self):
        """
        Find the original input size of the image or video.
        """
        if self.input_type == "single_image":
            return self.source.size
        elif self.input_type in ["video_dir", "video_file"]:
            return self.source.resolution
        elif self.input_type == "image_directory":
            return None
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")

class VideoDataset(Dataset):
    """Video dataset"""
    
    def __init__(self, video: Video, transform: nn.Module = None):
        """
        PyTorch dataset for video frames.
        Args:
            video (Video): Video object containing frames.
            transform (nn.Module): Transform to apply to each frame.
        """
        self.video = video
        self.transform = transform if transform is not None else nn.Identity()

    def __getitem__(self, idx):
        frame = self.video[idx]
        img = self.transform(frame)
        return {
            "img": img, 
            "frame_idx": idx,
        }

    def __len__(self):
        return len(self.video)

class ImageDirectoryDataset(Dataset):
    """Dataset for images in a directory."""

    def __init__(self,
                 image_directory: ImageDirectory,
                 transform_factory: TransformFactory,
                 resize_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            image_directory (ImageDirectory): Directory containing images.
            transform_factory (TransformFactory): Factory to create transforms for images.
            resize_size (Optional[Tuple[int, int]]): Size to resize images to.
        """
        self.image_directory = image_directory
        self.transform_factory = transform_factory
        self.resize_size = resize_size
    
    def __getitem__(self, idx):
        img = self.image_directory[idx]
        if self.resize_size is not None:
            transform = self.transform_factory.get_transform(self.resize_size)
        else:
            transform = self.transform_factory.get_transform(img.size)
        img_tensor = transform.transform(img)
        return {
            "img": img_tensor,
            "filename": self.image_directory.image_names[idx],
            "feature_map_size": transform.feature_map_size,
        }

    def __len__(self):
        return len(self.image_directory)


def create_xarray_from_batch_frames(batch_frames: List[FrameData]) -> xr.DataArray:
    """
    Create xarray from batch frames.
    """
    # Check if all frames have the same feature dimensions
    feature_shapes = [frame.features.shape for frame in batch_frames]
    if len(set(feature_shapes)) > 1:
        raise ValueError(f"Cannot create xarray from frames with different feature shapes: {set(feature_shapes)}")
    
    tensor = torch.stack([x.features for x in batch_frames])
    frame_idx = [x.frame_idx for x in batch_frames]
    
    # Assuming the tensor has shape (batch, height, width, feature)
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


def create_dataframe_from_batch_frames(batch_frames: List[FrameData]) -> pd.DataFrame:
    """
    Create a DataFrame from batch frames, handling varying sizes.
    """
    # Check if all frames have the same feature dimensions
    feature_shapes = [frame.features.shape for frame in batch_frames]
    if len(set(feature_shapes)) > 1:
        # Handle varying sizes by processing each frame separately
        dfs = []
        for frame in batch_frames:
            df = create_dataframe_from_single_frame(frame)
            dfs.append(df)
        return pd.concat(dfs, axis=0)
    
    # All frames have the same shape - use original logic
    tensor = torch.stack([x.features for x in batch_frames])
    frame_idx_set = [x.frame_idx for x in batch_frames]

    n_patches = tensor.shape[1] * tensor.shape[2]

    features = rearrange(tensor, "b h w f -> (b h w) f").cpu().numpy()

    frame_idx = []
    patch_idx = []
    for idx in frame_idx_set:
        frame_idx.extend([int(idx)] * n_patches)
        patch_idx.extend(list(range(n_patches)))

    # patch_idx
    index = pd.MultiIndex.from_tuples(
        list(zip(frame_idx, patch_idx)), names=["frame_idx", "patch_idx"]
    )

    columns = [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, index=index, columns=columns)
    return df


def create_dataframe_from_single_frame(frame: FrameData) -> pd.DataFrame:
    """Create DataFrame from a single frame."""
    if frame.flattened:
        features = frame.features.cpu().numpy()
        n_patches = features.shape[0]
    else:
        features = rearrange(frame.features, "h w f -> (h w) f").cpu().numpy()
        n_patches = features.shape[0]
    
    frame_idx = [frame.frame_idx] * n_patches
    patch_idx = list(range(n_patches))
    
    index = pd.MultiIndex.from_tuples(
        list(zip(frame_idx, patch_idx)), names=["frame_idx", "patch_idx"]
    )
    
    columns = [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, index=index, columns=columns)
    
    return df