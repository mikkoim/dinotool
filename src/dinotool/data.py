import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Union, Literal
import cv2
from dataclasses import dataclass
import numpy as np
from torchvision import transforms
import xarray as xr
from einops import rearrange
import pandas as pd
from collections import defaultdict

import torch
from einops import rearrange


class LocalFeatures:
    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        is_flattened=False,
        is_normalized=False,
        h=None,
        w=None,
    ):
        """
        tensor: torch.Tensor of shape (b, h, w, f) or (b, h*w, f)
        """
        self.tensor = tensor
        self.is_flattened = is_flattened
        self.is_normalized = is_normalized

        shape = tensor.shape
        if not is_flattened and len(shape) == 4:
            self.b, self.h, self.w, self.f = shape
        elif is_flattened and len(shape) == 3:
            self.b, hw, self.f = shape
            if h is None or w is None:
                raise ValueError("Flattened input requires h and w.")
            self.h, self.w = h, w
        else:
            raise ValueError("Unexpected tensor shape for LocalFeatures.")

    def flat(self):
        """Return a new LocalFeatures instance with (b, h*w, f) shape."""
        if self.is_flattened:
            return self
        flat_tensor = rearrange(self.tensor, "b h w f -> b (h w) f")
        return LocalFeatures(
            flat_tensor,
            is_flattened=True,
            is_normalized=self.is_normalized,
            h=self.h,
            w=self.w,
        )

    def full(self):
        """Return a new LocalFeatures instance with (b, h, w, f) shape."""
        if not self.is_flattened:
            return self
        full_tensor = rearrange(self.tensor, "b (h w) f -> b h w f", h=self.h, w=self.w)
        return LocalFeatures(
            full_tensor, is_flattened=False, is_normalized=self.is_normalized
        )

    def normalize(self, eps=1e-8):
        """L2-normalize along feature dimension."""
        normed = torch.nn.functional.normalize(self.tensor, p=2, dim=-1, eps=eps)
        return LocalFeatures(
            normed,
            is_flattened=self.is_flattened,
            is_normalized=True,
            h=self.h,
            w=self.w,
        )

    def clone(self):
        """Clone the underlying tensor and metadata."""
        return LocalFeatures(
            self.tensor.clone(),
            is_flattened=self.is_flattened,
            is_normalized=self.is_normalized,
            h=self.h,
            w=self.w,
        )

    def to(self, *args, **kwargs):
        """Move tensor to device/dtype."""
        return LocalFeatures(
            self.tensor.to(*args, **kwargs),
            is_flattened=self.is_flattened,
            is_normalized=self.is_normalized,
            h=self.h,
            w=self.w,
        )

    def __repr__(self):
        layout = "flat" if self.is_flattened else "full"
        norm = "normalized" if self.is_normalized else "unnormalized"
        return f"LocalFeatures({layout}, {norm}, shape={tuple(self.tensor.shape)})"

    def __getitem__(self, index):
        """Index the batch dimension and return a new LocalFeatures object."""
        indexed_tensor = self.tensor[index]
        return LocalFeatures(
            indexed_tensor.unsqueeze(0),  # Add batch dim back
            is_flattened=self.is_flattened,
            is_normalized=self.is_normalized,
            h=self.h,
            w=self.w,
        )

    @property
    def shape(self):
        return self.tensor.shape


@dataclass
class FrameData:
    img: Image.Image
    features: LocalFeatures
    pca: Optional[np.ndarray] = None
    frame_idx: Optional[int] = None
    filename: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.features, LocalFeatures):
            raise TypeError("features must be an instance of LocalFeatures")
        if not isinstance(self.img, Image.Image):
            raise TypeError("img must be an instance of PIL.Image.Image")

        # Ensure either frame_idx or filename is provided
        if self.frame_idx is None and self.filename is None:
            raise ValueError("Either frame_idx or filename must be provided.")
        # Ensure only one of them is provided
        if self.frame_idx is not None and self.filename is not None:
            raise ValueError("Cannot provide both frame_idx and filename. Choose one.")
        if self.frame_idx is not None and not isinstance(self.frame_idx, int):
            raise TypeError("frame_idx must be an integer or None.")
        if self.filename is not None and not isinstance(self.filename, str):
            raise TypeError("filename must be a string or None.")

        if self.pca is not None and not isinstance(self.pca, np.ndarray):
            raise TypeError("pca must be a numpy ndarray or None")

        if not self.features.is_normalized:
            raise ValueError(
                "Features must be normalized. Use features.normalize() to normalize them."
            )

        if self.features.is_flattened:
            self.features = self.features.full()

        if len(self.features.tensor.shape) != 4:
            raise ValueError(
                "Features tensor must have 4 dimensions (b, h, w, f) after full() call."
            )
        if self.features.tensor.shape[0] != 1:
            raise ValueError(
                "Features tensor must have batch size of 1 after full() call."
            )


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
            p
            for p in os.listdir(path)
            if os.path.splitext(p)[-1].lower()
            in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
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
        if hasattr(self, "video_capture"):
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
            p
            for p in os.listdir(path)
            if os.path.splitext(p)[-1].lower()
            in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
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

    def __init__(self, model_name, patch_size: int) -> nn.Module:
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
        dummy_transformed = transform(
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        )
        resize_size = (dummy_transformed.shape[2], dummy_transformed.shape[1])

        dims = calculate_dino_dimensions(resize_size, patch_size=self.patch_size)
        model_input_size = (dims["w"], dims["h"])
        feature_map_size = (dims["w_featmap"], dims["h_featmap"])

        self.transform = OpenCLIPTransform(
            transform=transform,
            resize_size=model_input_size,
            feature_map_size=feature_map_size,
        )
        return self.transform

    def get_dino_transform(self, input_size: Tuple[int, int]) -> nn.Module:
        if input_size in self._transform_cache:
            # If a transform for this input size is already cached, return it
            return self._transform_cache[input_size]

        dims = calculate_dino_dimensions(input_size, patch_size=self.patch_size)
        model_input_size = (dims["w"], dims["h"])
        feature_map_size = (dims["w_featmap"], dims["h_featmap"])

        transform = transforms.Compose(
            [
                transforms.Resize((model_input_size[1], model_input_size[0])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform = DINOTransform(
            transform=transform,
            resize_size=model_input_size,
            feature_map_size=feature_map_size,
        )
        self._transform_cache[input_size] = self.transform
        return self.transform

    def get_transform(self, input_size: Tuple[int, int]) -> nn.Module:
        if self.model_type == "openclip":
            return self.get_openclip_transform()
        elif self.model_type == "dino":
            return self.get_dino_transform(input_size)


@dataclass
class InputData:
    """
    Data class to hold input data for feature extraction.
    This class is used to store the source image or video, the transformed data,
    and the input and feature map sizes.
    """

    source: Union[Image.Image, Video, ImageDirectory]
    data: Union[torch.Tensor, DataLoader]
    input_size: Optional[Tuple[int, int]] = None
    feature_map_size: Optional[Tuple[int, int]] = None
    input_type: str = "unknown"


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

    def __init__(
        self,
        model_name: str,
        input_path: str,
        patch_size: int,
        batch_size: int = 1,
        resize_size: Optional[Tuple[int, int]] = None,
    ):
        self.model_name = model_name
        self.input_path = input_path
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.resize_size = resize_size

        self.source, self.input_type = self.find_source(input_path)

    @staticmethod
    def find_source(input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        if os.path.isdir(input_path):
            try:
                source = Video(input_path)
                return source, "video_dir"
            except Exception as video_error:
                try:
                    source = ImageDirectory(input_path)
                    return source, "image_directory"
                except Exception as image_error:
                    raise ValueError(
                        f"Directory '{input_path}' could not be processed as a video or image directory.\n"
                        f"--> Video-related error: {video_error}\n"
                        f"--> Image-directory-related error: {image_error}"
                    ) from image_error
        elif os.path.isfile(input_path):
            try:
                source = Image.open(input_path).convert("RGB")
                return source, "single_image"
            except Image.UnidentifiedImageError:
                try:
                    source = Video(input_path)
                    return source, "video_file"
                except Exception as video_error:
                    raise ValueError(
                        f"File '{input_path}' could not be identified as a supported image or video file.\n"
                        f"--> Video-related error: {video_error}"
                    ) from video_error
        else:
            raise ValueError(
                f"Input path '{input_path}' is not a valid file or directory."
            )

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
            model_name=self.model_name, patch_size=self.patch_size
        )
        if self.resize_size is not None:
            print(f"Resizing all input to {self.resize_size}")
        ds = ImageDirectoryDataset(
            self.source,
            transform_factory=transform_factory,
            resize_size=self.resize_size,
        )
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        return InputData(
            source=self.source,
            data=dataloader,
            input_size=None,  # Varying size, so no fixed input size
            feature_map_size=None,  # Varying size, so no fixed feature map size
            input_type=self.input_type,
        )

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
            model_name=self.model_name, patch_size=self.patch_size
        )
        self.transform = transform_factory.get_transform(original_input_size)
        print(f"Model input size: {self.transform.resize_size}")
        print(f"Feature map size: {self.transform.feature_map_size}")

        if self.input_type == "single_image":
            img_tensor = self.transform.transform(self.source).unsqueeze(0)
            return InputData(
                source=self.source,
                data=img_tensor,
                input_size=self.transform.resize_size,
                feature_map_size=self.transform.feature_map_size,
                input_type=self.input_type,
            )
        elif self.input_type in ["video_dir", "video_file"]:
            ds = VideoDataset(self.source, transform=self.transform.transform)
            dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            return InputData(
                source=self.source,
                data=dataloader,
                input_size=self.transform.resize_size,
                feature_map_size=self.transform.feature_map_size,
                input_type=self.input_type,
            )
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

    def __init__(
        self,
        image_directory: ImageDirectory,
        transform_factory: TransformFactory,
        resize_size: Optional[Tuple[int, int]] = None,
    ):
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


def create_xarray_from_batch_frames(
    batch_frames: List[FrameData], identifier: Literal["filename", "frame_idx"]
) -> xr.DataArray:
    """
    Create xarray from batch frames.
    """
    # Check if all frames have the same feature dimensions
    feature_shapes = [frame.features.shape for frame in batch_frames]
    if len(set(feature_shapes)) > 1:
        raise ValueError(
            f"Cannot create xarray from frames with different feature shapes: {set(feature_shapes)}"
        )

    tensor = torch.cat([x.features.full().tensor for x in batch_frames], dim=0)
    identifier_list = [getattr(x, identifier) for x in batch_frames]

    # Assuming the tensor has shape (batch, height, width, feature)
    batch, height, width, feature = tensor.shape

    coords = {
        identifier: identifier_list,
        "y": np.arange(height),
        "x": np.arange(width),
        "feature": np.arange(feature),
    }
    data = xr.DataArray(
        tensor.cpu().numpy(),
        dims=(identifier, "y", "x", "feature"),
        coords=coords,
    )
    return data


def create_dataframe_from_batch_frames(
    batch_frames: List[FrameData], identifier: Literal["filename", "frame_idx"]
) -> pd.DataFrame:
    """
    Create a DataFrame from batch frames.
    """

    tensor = (
        torch.cat([x.features.flat().tensor for x in batch_frames], dim=0).cpu().numpy()
    )
    identifier_list = [getattr(x, identifier) for x in batch_frames]

    features = rearrange(tensor, "b hw f -> (b hw) f")

    n_patches = batch_frames[0].features.flat().shape[1]

    identifier_idx = []
    patch_idx = []

    for frame_identifier in identifier_list:
        identifier_idx.extend([frame_identifier] * n_patches)
        patch_idx.extend(list(range(n_patches)))

    # patch_idx
    index = pd.MultiIndex.from_tuples(
        list(zip(identifier_idx, patch_idx)), names=[identifier, "patch_idx"]
    )

    columns = [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, index=index, columns=columns)
    return df
