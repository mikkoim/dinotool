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


def run_backbone_all_test(name, input_size=None):
    out_path = f"test/outputs/backbones/{name}_all"
    model_name = MODEL_SHORTCUTS[name]
    config = DinotoolConfig(
        model_name=model_name,
        input="test/data/magpie.jpg",
        output=f"{out_path}.jpg",
        save_features="all",
        input_size=input_size,
        no_vis=True,
    )
    DinotoolProcessor(config).run()
    assert os.path.exists(f"{out_path}.parquet")
    assert os.path.exists(f"{out_path}.txt")
    global_df = pd.read_csv(f"{out_path}.txt", header=None)
    assert global_df.shape[0] == 1
    assert np.allclose(np.linalg.norm(global_df.values), 1.0, atol=1e-5)

def test_all_mode_dinov2():  run_backbone_all_test("vit-s",    input_size=(64, 64))
def test_all_mode_dinov3():  run_backbone_all_test("dinov3-s", input_size=(64, 64))
def test_all_mode_siglip():  run_backbone_all_test("siglip2",  input_size=(64, 64))
def test_all_mode_clip():    run_backbone_all_test("clip",     input_size=(64, 64))
def test_all_mode_radio():   run_backbone_all_test("radio-b",  input_size=(64, 64))


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


def test_image_features_all():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/out_all.jpg",
        save_features="all",
    )
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists("test/outputs/out_all.jpg")
    assert os.path.exists("test/outputs/out_all.parquet")
    assert os.path.exists("test/outputs/out_all.txt")

    df = pd.read_parquet("test/outputs/out_all.parquet")
    assert df.shape == (910, 384)
    assert df.index.names == ["frame_idx", "patch_idx"]

    global_df = pd.read_csv("test/outputs/out_all.txt", header=None)
    assert global_df.shape == (1, 384)
    assert np.allclose(np.linalg.norm(global_df.values), 1.0, atol=1e-5)


def test_image_no_vis_flat():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/novis_flat",
        save_features="flat",
        no_vis=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/novis_flat.parquet")
    assert not os.path.exists("test/outputs/novis_flat.jpg")

    df = pd.read_parquet("test/outputs/novis_flat.parquet")
    assert df.shape == (910, 384)
    assert np.allclose(np.linalg.norm(df.values, axis=1), 1.0, atol=1e-5)


def test_image_no_vis_frame():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/novis_frame",
        save_features="frame",
        no_vis=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/novis_frame.txt")


def test_image_only_pca():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/only_pca.jpg",
        only_pca=True,
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/only_pca.jpg")
    from PIL import Image
    img = Image.open("test/outputs/only_pca.jpg")
    # only_pca produces just the PCA image, not the stacked original+PCA
    w, h = img.size
    # Should be a single image (not double height from stacking)
    assert h < 400  # original magpie is ~370px tall; stacked would be ~740


def test_image_with_input_size():
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output="test/outputs/resized.jpg",
        save_features="full",
        input_size=(224, 224),
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/resized.jpg")
    assert os.path.exists("test/outputs/resized.nc")

    ds = xr.open_dataarray("test/outputs/resized.nc")
    # 224/14 = 16 patches in each dimension
    assert len(ds.y) == 16
    assert len(ds.x) == 16
    assert len(ds.feature) == 384


def test_image_png_input():
    config = DinotoolConfig(
        input="test/data/pepper.png",
        output="test/outputs/pepper_out.jpg",
        save_features="full",
    )
    processor = DinotoolProcessor(config)
    processor.run()

    assert os.path.exists("test/outputs/pepper_out.jpg")
    assert os.path.exists("test/outputs/pepper_out.nc")

    ds = xr.open_dataarray("test/outputs/pepper_out.nc")
    assert len(ds.frame_idx) == 1
    assert len(ds.feature) == 384
    assert np.allclose(
        np.linalg.norm(ds.sel(x=0, y=0, frame_idx=0).values), 1.0, atol=1e-5
    )
