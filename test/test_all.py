from dinotool.cli import main, DinotoolConfig
from pathlib import Path
import os

Path("test/outputs").mkdir(exist_ok=True)


def test_full_image():
    args = DinotoolConfig(input="test/data/magpie.jpg", output="test/outputs/out.jpg")

    main(args)
    assert os.path.exists("test/outputs/out.jpg")


def test_full_image_features():
    args = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/out.jpg",
        save_features="full",
    )

    main(args)
    assert os.path.exists("test/outputs/out.jpg")
    assert os.path.exists("test/outputs/out.nc")
    import xarray as xr

    ds = xr.open_dataarray("test/outputs/out.nc")
    assert len(ds.frame_idx) == 1
    assert len(ds.y) == 26
    assert len(ds.x) == 35
    assert len(ds.feature) == 384

def test_full_image_features_siglip2():
    args = DinotoolConfig(
        model_name="hf-hub:timm/ViT-B-16-SigLIP2-512",
        input="test/data/magpie.jpg",
        output="test/outputs/out-siglip2.jpg",
        save_features="full",
    )

    main(args)
    assert os.path.exists("test/outputs/out-siglip2.jpg")
    assert os.path.exists("test/outputs/out-siglip2.nc")
    import xarray as xr

    ds = xr.open_dataarray("test/outputs/out-siglip2.nc")
    assert len(ds.frame_idx) == 1
    assert len(ds.y) == 32
    assert len(ds.x) == 32
    assert len(ds.feature) == 768


def test_full_image_features_flat():
    args = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/out.jpg",
        save_features="flat",
    )

    main(args)
    assert os.path.exists("test/outputs/out.jpg")
    assert os.path.exists("test/outputs/out.parquet")
    import pandas as pd

    df = pd.read_parquet("test/outputs/out.parquet")
    assert df.shape == (910, 384)


def test_full_image_features_frame():
    args = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/out",
        save_features="frame",
    )

    main(args)
    assert os.path.exists("test/outputs/out.txt")
    import pandas as pd

    df = pd.read_csv("test/outputs/out.txt", header=None)
    assert df.shape == (1, 384)


def test_full_video_file():
    args = DinotoolConfig(
        input="test/data/nasa.mp4", output="test/outputs/nasaout1.mp4", batch_size=4
    )

    main(args)
    assert os.path.exists("test/outputs/nasaout1.mp4")


def test_full_video_folder():
    args = DinotoolConfig(
        input="test/data/nasa_frames", output="test/outputs/nasaout2.mp4", batch_size=4
    )

    main(args)
    assert os.path.exists("test/outputs/nasaout2.mp4")


def test_full_video_file_features():
    args = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasaout3.mp4",
        batch_size=4,
        save_features="full",
    )

    main(args)
    import xarray as xr

    assert os.path.exists("test/outputs/nasaout3.zarr")
    ds = xr.open_dataarray("test/outputs/nasaout3.zarr")
    assert len(ds.frame_idx) == 90
    assert len(ds.y) == 19
    assert len(ds.x) == 34
    assert len(ds.feature) == 384


def test_full_video_folder_features():
    args = DinotoolConfig(
        input="test/data/nasa_frames",
        output="test/outputs/nasaout4.mp4",
        batch_size=4,
        save_features="full",
    )

    main(args)
    import xarray as xr

    assert os.path.exists("test/outputs/nasaout4.zarr")
    ds = xr.open_dataarray("test/outputs/nasaout4.zarr")
    assert len(ds.frame_idx) == 90
    assert len(ds.y) == 19
    assert len(ds.x) == 34
    assert len(ds.feature) == 384


def test_full_video_file_features_flat():
    args = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasaout5.mp4",
        batch_size=4,
        save_features="flat",
    )

    main(args)
    import dask.dataframe as dd

    assert os.path.exists("test/outputs/nasaout5.parquet")
    df = dd.read_parquet("test/outputs/nasaout5.parquet")


def test_full_video_file_features_frame():
    args = DinotoolConfig(
        input="test/data/nasa.mp4",
        output="test/outputs/nasaout6.mp4",
        batch_size=4,
        save_features="frame",
    )

    main(args)
    import pandas as pd

    df = pd.read_parquet("test/outputs/nasaout6.parquet")
    assert df.shape == (90, 384)


def test_full_video_folder_features_flat():
    args = DinotoolConfig(
        input="test/data/nasa_frames",
        output="test/outputs/nasaout7.mp4",
        batch_size=4,
        save_features="flat",
    )

    main(args)
    import dask.dataframe as dd

    assert os.path.exists("test/outputs/nasaout7.parquet")
    df = dd.read_parquet("test/outputs/nasaout7.parquet")
