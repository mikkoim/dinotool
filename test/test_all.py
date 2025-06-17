from dinotool.cli import DinotoolConfig, DinotoolProcessor
from pathlib import Path
import os
import pandas as pd
import xarray as xr

Path("test/outputs").mkdir(exist_ok=True)


def test_full_image():
    config = DinotoolConfig(input="test/data/magpie.jpg", output="test/outputs/out.jpg")
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/out.jpg")


def test_full_image_features():
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

def test_full_image_features_siglip2():
    config = DinotoolConfig(
        model_name="hf-hub:timm/ViT-B-16-SigLIP2-512",
        input="test/data/magpie.jpg",
        output="test/outputs/out-siglip2.jpg",
        save_features="full",
    )
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/out-siglip2.jpg")
    assert os.path.exists("test/outputs/out-siglip2.nc")

    ds = xr.open_dataarray("test/outputs/out-siglip2.nc")
    assert len(ds.frame_idx) == 1
    assert len(ds.y) == 32
    assert len(ds.x) == 32
    assert len(ds.feature) == 768

def test_full_image_features_flat():
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
    assert df.index.names == ['frame_idx', 'patch_idx']
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]

def test_full_image_features_frame():
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


def test_full_video_file():
    config = DinotoolConfig(
        input="test/data/nasa.mp4", output="test/outputs/nasaout1.mp4", batch_size=4
    )

    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/nasaout1.mp4")


def test_full_video_folder():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small", output="test/outputs/nasaout2.mp4", batch_size=4
    )

    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/nasaout2.mp4")


def test_full_video_file_features():
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


def test_full_video_folder_features():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasaout4.mp4",
        batch_size=4,
        save_features="full",
    )

    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasaout4.zarr")
    ds = xr.open_dataarray("test/outputs/nasaout4.zarr")
    assert len(ds.frame_idx) == 9
    assert len(ds.y) == 19
    assert len(ds.x) == 34
    assert len(ds.feature) == 384

def test_full_video_folder_new_folder():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small", output="test/outputs/testfolder/nasaout1.mp4", batch_size=4
    )

    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/testfolder/nasaout1.mp4")


def test_full_video_file_features_flat():
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
    breakpoint()
    assert df.shape == (58140, 384)
    assert df.index.names == ['frame_idx', 'patch_idx']
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]


def test_full_video_file_features_frame():
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


def test_full_video_folder_features_flat():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasaout7.mp4",
        batch_size=4,
        save_features="flat",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasaout7.parquet")
    df = pd.read_parquet("test/outputs/nasaout7.parquet")
    assert df.shape == (5814, 384)
    assert df.index.names == ['frame_idx', 'patch_idx']
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]

def test_full_video_folder_features_flat_no_vis():
    config = DinotoolConfig(
        input="test/data/nasa_frames_small",
        output="test/outputs/nasaout7_novis.mp4",
        batch_size=4,
        save_features="flat",
        no_vis = True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/nasaout7_novis.parquet")
    df = pd.read_parquet("test/outputs/nasaout7_novis.parquet")
    assert df.shape == (5814, 384)
    assert df.index.names == ['frame_idx', 'patch_idx']
    assert df.columns.tolist() == [f"feature_{i}" for i in range(384)]

def test_full_imagedir():
    config = DinotoolConfig(
        input="test/data/imagefolder",
        output="test/outputs/if1"
    )
    processor = DinotoolProcessor(config)
    processor.run()

    output_dir = Path("test/outputs/if1")
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.jpg"))) == 4

def test_full_imagedir_features_full():
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

def test_full_imagedir_features_flat():
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

def test_full_imagedir_features_frame():
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
