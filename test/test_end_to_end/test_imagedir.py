from dinotool.cli import DinotoolConfig, DinotoolProcessor
from pathlib import Path
import os
import pandas as pd
import numpy as np
import xarray as xr


def test_imagedir_only():
    config = DinotoolConfig(input="test/data/imagefolder", output="test/outputs/if1")
    processor = DinotoolProcessor(config)
    processor.run()

    output_dir = Path("test/outputs/if1")
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.jpg"))) == 4


def test_imagedir_features_full():
    config = DinotoolConfig(
        input="test/data/imagefolder", output="test/outputs/if1", save_features="full"
    )
    processor = DinotoolProcessor(config)
    processor.run()

    output_dir = Path("test/outputs/if1")
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.jpg"))) == 4
    assert len(list(output_dir.glob("*"))) == 8

    ds = xr.open_dataarray("test/outputs/if1/bird1.nc")
    assert len(ds.frame_idx) == 1
    assert len(ds.y) == 64
    assert len(ds.x) == 64
    assert len(ds.feature) == 384
    assert np.allclose(
        np.linalg.norm(ds.sel(x=0, y=0, frame_idx=0).values), 1.0, atol=1e-5
    )


def test_imagedir_features_flat():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1_flat",
        save_features="flat",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    output_dir = Path("test/outputs/if1_flat")
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.jpg"))) == 4
    assert len(list(output_dir.glob("*"))) == 8

    df = pd.read_parquet("test/outputs/if1_flat/bird1.parquet")
    assert df.shape == (4096, 384)
    assert df.index.names == ["frame_idx", "patch_idx"]
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_imagedir_features_frame():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1_frame",
        save_features="frame",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    df = pd.read_parquet("test/outputs/if1_frame.parquet")
    assert df.shape == (4, 384)
    assert df.index.names == ["filename"]
    assert set(df.index) == set(
        [x.name for x in Path("test/data/imagefolder").glob("*")]
    )
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


# Resized and batch processed
def test_batched_imagedir_features_full():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1_b",
        save_features="full",
        batch_size=2,
        input_size=(480, 270),
        no_vis=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    ds = xr.open_zarr("test/outputs/if1_b.zarr").to_dataarray()

    assert len(ds.filename) == 4
    assert len(ds.y) == 19
    assert len(ds.x) == 34
    assert len(ds.feature) == 384
    assert np.allclose(
        np.linalg.norm(ds.sel(x=0, y=0, filename="bird1.jpg").values), 1.0, atol=1e-5
    )


def test_batched_imagedir_features_flat():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1_flat_b",
        save_features="flat",
        batch_size=2,
        input_size=(480, 270),
        no_vis=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    df = pd.read_parquet("test/outputs/if1_flat_b.parquet")
    assert df.shape == (2584, 384)
    assert df.index.names == ["filename", "patch_idx"]
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_batched_imagedir_features_frame():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1_frame_b",
        save_features="frame",
        batch_size=2,
        input_size=(480, 270),
        no_vis=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    df = pd.read_parquet("test/outputs/if1_frame_b.parquet")
    assert df.shape == (4, 384)
    assert df.index.names == ["filename"]
    assert set(df.index) == set(
        [x.name for x in Path("test/data/imagefolder").glob("*")]
    )
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)
