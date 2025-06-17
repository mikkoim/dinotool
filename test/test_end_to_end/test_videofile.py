from dinotool.cli import DinotoolConfig, DinotoolProcessor
from pathlib import Path
import os
import pandas as pd
import xarray as xr

def test_videofile_only():
    config = DinotoolConfig(
        input="test/data/nasa.mp4", output="test/outputs/nasaout1.mp4", batch_size=4
    )

    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/nasaout1.mp4")

def test_videofile_features_full():
    config = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasaout3.mp4",
        batch_size=4,
        save_features="full",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasaout3.zarr")
    ds = xr.open_dataarray("test/outputs/nasaout3.zarr")
    assert len(ds.frame_idx) == 90
    assert len(ds.y) == 19
    assert len(ds.x) == 34
    assert len(ds.feature) == 384

def test_videofile_features_flat():
    config = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasaout5.mp4",
        batch_size=4,
        save_features="flat",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasaout5.parquet")
    df = pd.read_parquet("test/outputs/nasaout5.parquet")

    assert df.shape == (58140, 384)
    assert df.index.names == ['frame_idx', 'patch_idx']
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]


def test_videofile_features_frame():
    config = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasaout6.mp4",
        batch_size=4,
        save_features="frame",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasaout6.parquet")
    df = pd.read_parquet("test/outputs/nasaout6.parquet")
    assert df.shape == (90, 384)
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]