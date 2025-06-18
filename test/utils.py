import pytest
from dinotool.data import Video, VideoDataset, FrameData
from dinotool.model import load_model
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image


def setup_model_and_batch():
    model = load_model("dinov2_vits14_reg")
    video = Video("test/data/nasa.mp4")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = VideoDataset(video, transform=transform)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    return {"model": model, "batch": batch}
