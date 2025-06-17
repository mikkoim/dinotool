from dinotool.cli import DinotoolConfig, DinotoolProcessor
from pathlib import Path
import os
import pandas as pd
import xarray as xr

def test_imagedir_only():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1"
    )
    processor = DinotoolProcessor(config)
    processor.run()

    output_dir = Path("test/outputs/if1")
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.jpg"))) == 4

def test_imagedir_features_full():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1",
        save_features="full"
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

def test_imagedir_features_flat():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1_flat",
        save_features="flat"
    )
    processor = DinotoolProcessor(config)
    processor.run()

    output_dir = Path("test/outputs/if1_flat")
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.jpg"))) == 4
    assert len(list(output_dir.glob("*"))) == 8

    df = pd.read_parquet("test/outputs/if1_flat/bird1.parquet")
    assert df.shape == (4096, 384)
    assert df.index.names == ['frame_idx', 'patch_idx']
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]

def test_imagedir_features_frame():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1_frame",
        save_features="frame"
    )
    processor = DinotoolProcessor(config)
    processor.run()

    df = pd.read_parquet("test/outputs/if1_frame.parquet")
    assert df.shape == (4, 384)
    assert df.index.names == ['filename']
    assert set(df.index) == set([x.name for x in Path("test/data/imagefolder").glob("*")])
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]