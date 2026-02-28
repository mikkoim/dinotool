from dinotool.cli import DinotoolConfig, DinotoolProcessor
from pathlib import Path
import os
import pandas as pd
import numpy as np
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
    ds = xr.open_zarr("test/outputs/nasaout3.zarr").to_dataarray()
    assert len(ds.frame_idx) == 90
    assert len(ds.y) == 19
    assert len(ds.x) == 34
    assert len(ds.feature) == 384
    assert np.allclose(
        np.linalg.norm(ds.sel(x=0, y=0, frame_idx=0).values), 1.0, atol=1e-5
    )


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
    assert df.index.names == ["frame_idx", "patch_idx"]
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


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
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_videofile_no_vis():
    config = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasa_novis.mp4",
        batch_size=4,
        save_features="flat",
        no_vis=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasa_novis.parquet")
    # No video output should be created when no_vis is set
    assert not os.path.exists("test/outputs/nasa_novis.mp4")

    df = pd.read_parquet("test/outputs/nasa_novis.parquet")
    assert df.shape == (58140, 384)
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_videofile_only_pca():
    config = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasa_pca.mp4",
        batch_size=4,
        only_pca=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/nasa_pca.mp4")


def test_videofile_with_input_size():
    config = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasa_resized.mp4",
        batch_size=4,
        save_features="frame",
        input_size=(224, 224),
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasa_resized.parquet")
    df = pd.read_parquet("test/outputs/nasa_resized.parquet")
    assert df.shape == (90, 384)
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_videofile_batch_size_1():
    config = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasa_bs1.mp4",
        batch_size=1,
        save_features="frame",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasa_bs1.parquet")
    df = pd.read_parquet("test/outputs/nasa_bs1.parquet")
    assert df.shape == (90, 384)
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_videofile_avi_format():
    config = DinotoolConfig(
        input="test/data/sintel.avi",
        output="test/outputs/sintel_avi.mp4",
        batch_size=4,
        save_features="frame",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/sintel_avi.parquet")
    df = pd.read_parquet("test/outputs/sintel_avi.parquet")
    assert df.shape[1] == 384
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)
