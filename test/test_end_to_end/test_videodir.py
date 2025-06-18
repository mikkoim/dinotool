from dinotool.cli import DinotoolConfig, DinotoolProcessor
from pathlib import Path
import os
import pandas as pd
import numpy as np
import xarray as xr


def test_videodir_only():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasa_videodir_only.mp4",
        batch_size=4,
    )

    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/nasa_videodir_only.mp4")


def test_videodir_newfolder():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/testfolder/nasa_videodir_newfolder.mp4",
        batch_size=4,
    )

    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/testfolder/nasa_videodir_newfolder.mp4")


def test_videodir_features_full():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasa_videodir_features_full.mp4",
        batch_size=4,
        save_features="full",
    )

    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasa_videodir_features_full.zarr")
    ds = xr.open_dataarray("test/outputs/nasa_videodir_features_full.zarr")
    assert len(ds.frame_idx) == 9
    assert len(ds.y) == 19
    assert len(ds.x) == 34
    assert len(ds.feature) == 384
    assert np.allclose(
        np.linalg.norm(ds.sel(x=0, y=0, frame_idx=0).values), 1.0, atol=1e-5
    )


def test_videodir_features_flat():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasa_videodir_features_flat.mp4",
        batch_size=4,
        save_features="flat",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasa_videodir_features_flat.parquet")
    df = pd.read_parquet("test/outputs/nasa_videodir_features_flat.parquet")
    assert df.shape == (5814, 384)
    assert df.index.names == ["frame_idx", "patch_idx"]
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_videodir_features_flat_novis():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasa_videodir_features_flat_novis.mp4",
        batch_size=4,
        save_features="flat",
        no_vis=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasa_videodir_features_flat_novis.parquet")
    df = pd.read_parquet("test/outputs/nasa_videodir_features_flat_novis.parquet")
    assert df.shape == (5814, 384)
    assert df.index.names == ["frame_idx", "patch_idx"]
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_videodir_features_frame():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasa_videodir_features_frame.mp4",
        batch_size=4,
        save_features="frame",
    )

    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasa_videodir_features_frame.parquet")
    df = pd.read_parquet("test/outputs/nasa_videodir_features_frame.parquet")

    assert df.shape == (9, 384)
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)
