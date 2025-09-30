#export HF_HUB_CACHE=/mnt/d/models
#export TORCH_HOME=/mnt/d/models
#export CUDA_VISIBLE_DEVICES=

import pytest

pytestmark = pytest.mark.slow

from dinotool.cli import DinotoolConfig, DinotoolProcessor, MODEL_SHORTCUTS
import os
import numpy as np
import xarray as xr
from pathlib import Path

Path("test/outputs/backbones").mkdir(parents=True, exist_ok=True)

def assert_full_features(base_path, h, w, f):
    assert os.path.exists(f"{base_path}.jpg")
    assert os.path.exists(f"{base_path}.nc")

    ds = xr.open_dataarray(f"{base_path}.nc")
    print(ds.shape)
    assert len(ds.frame_idx) == 1
    assert len(ds.y) == h
    assert len(ds.x) == w
    assert len(ds.feature) == f
    assert np.allclose(
        np.linalg.norm(ds.sel(x=0, y=0, frame_idx=0).values), 1.0, atol=1e-5
    )

def run_backbone_test(name, h, w, f, input_size=None):
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
    assert_full_features(out_path, h, w, f)

# DINOv2 models
def test_image_features_full_vits():
    run_backbone_test("vit-s", 26, 35, 384)

def test_image_features_full_vitb():
    run_backbone_test("vit-b", 26, 35, 768)

def test_image_features_full_vitl():
    run_backbone_test("vit-l", 4, 4, 1024, input_size=(64, 64))

def test_image_features_full_vitg():
    run_backbone_test("vit-g", 4, 4, 1536, input_size=(64, 64))

# SigLIP models
def test_image_features_full_siglip1():
    run_backbone_test("siglip1", 16, 16, 768)

def test_image_features_full_siglip2():
    run_backbone_test("siglip2", 32, 32, 768)

def test_image_features_full_siglip2_so400m_384():
    run_backbone_test("siglip2-so400m-384", 24, 24, 1152)

def test_image_features_full_siglip2_so400m_512():
    run_backbone_test("siglip2-so400m-512", 32, 32, 1152)

def test_image_features_full_siglip2_b16_256():
    run_backbone_test("siglip2-b16-256", 16, 16, 768)

def test_image_features_full_siglip2_b16_512():
    run_backbone_test("siglip2-b16-512", 32, 32, 768)

def test_image_features_full_siglip2_b32_256():
    run_backbone_test("siglip2-b32-256", 8, 8, 768)

# CLIP model

def test_image_features_full_clip():
    run_backbone_test("clip", 14, 14, 768)

# DINOv3 models
def test_image_features_full_dinov3_s():
    run_backbone_test("dinov3-s", 23, 31, 384)

def test_image_features_full_dinov3_splus():
    run_backbone_test("dinov3-splus", 23, 31, 384)

def test_image_features_full_dinov3_b():
    run_backbone_test("dinov3-b", 23, 31, 768)

def test_image_features_full_dinov3_l():
    run_backbone_test("dinov3-l", 4, 4, 1024, input_size=(64, 64))

def test_image_features_full_dinov3_hplus():
    run_backbone_test("dinov3-hplus", 4, 4, 1280, input_size=(64, 64))

# def test_image_features_full_dinov3_7b():
#     run_backbone_test("dinov3-7b", 4, 4, 768, input_size=(64, 64))

def test_image_features_full_dinov3_lsat():
    run_backbone_test("dinov3-l-sat", 23, 31, 1024)

# def test_image_features_full_dinov3_7bsat():
#     run_backbone_test("dinov3-7b-sat", 4, 4, 768, input_size=(64, 64))


# AM-RADIO models
def test_image_features_full_radio_b():
    run_backbone_test("radio-b", 23, 31, 768)

def test_image_features_full_radio_l():
    run_backbone_test("radio-l", 23, 31, 1024)
    
def test_image_features_full_radio_h():
    run_backbone_test("radio-h", 23, 31, 1280)

def test_image_features_full_radio_g():
    run_backbone_test("radio-g", 23, 31, 1536)