from dinotool.cli import DinotoolConfig, DinotoolProcessor, MODEL_SHORTCUTS
from pathlib import Path
import os
import pandas as pd
import numpy as np
import xarray as xr
import pytest

def run_backbone_test(name, input_size=None):
    out_path = f"test/outputs/backbones/{name}"
    model_name = MODEL_SHORTCUTS[name]

    config = DinotoolConfig(
        model_name=model_name,
        input="test/data/magpie.jpg",
        output=f"{out_path}.jpg",
        save_features="full",
        input_size=input_size
    )
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists(f"{out_path}.jpg")

def test_smoke_dinov2():
    run_backbone_test("vit-s", input_size=(64,64))
def test_smoke_dinov3():
    run_backbone_test("dinov3-s", input_size=(64,64))
def test_smoke_siglip2():
    run_backbone_test("siglip2", input_size=(64,64))
def test_smoke_clip():
    run_backbone_test("clip", input_size=(64,64))
def test_smoke_radio():
    run_backbone_test("radio-b", input_size=(64,64))


def test_image_features_full():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/out.jpg",
        save_features="full",
    )
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/out.jpg")
    assert os.path.exists("test/outputs/out.nc")

    ds = xr.open_dataarray("test/outputs/out.nc")
    assert len(ds.frame_idx) == 1
    assert len(ds.y) == 26
    assert len(ds.x) == 35
    assert len(ds.feature) == 384
    assert np.allclose(
        np.linalg.norm(ds.sel(x=0, y=0, frame_idx=0).values), 1.0, atol=1e-5
    )



def test_image_features_flat():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/out.jpg",
        save_features="flat",
    )
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/out.jpg")
    assert os.path.exists("test/outputs/out.parquet")

    df = pd.read_parquet("test/outputs/out.parquet")
    assert df.shape == (910, 384)
    assert df.index.names == ["frame_idx", "patch_idx"]
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_image_features_frame():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/out",
        save_features="frame",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/out.txt")

    df = pd.read_csv("test/outputs/out.txt", header=None)
    assert df.shape == (1, 384)
    assert np.allclose(np.linalg.norm(df.values), 1.0, atol=1e-5)
