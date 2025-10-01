#export HF_HUB_CACHE=/mnt/d/models
#export TORCH_HOME=/mnt/d/models
#export CUDA_VISIBLE_DEVICES=

import pytest

pytestmark = pytest.mark.slow

from dinotool.cli import DinotoolConfig, DinotoolProcessor, MODEL_SHORTCUTS
import os
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

Path("test/outputs/backbones-global").mkdir(parents=True, exist_ok=True)

def run_backbone_test(name, f, input_size=None):
    out_path = f"test/outputs/backbones-global/{name}"
    model_name = MODEL_SHORTCUTS[name]

    config = DinotoolConfig(
        model_name=model_name,
        input="test/data/magpie.jpg",
        output=f"{out_path}",
        save_features="frame",
        input_size=input_size
    )
    processor = DinotoolProcessor(config)
    processor.run()

    df = pd.read_csv(f"{out_path}.txt", header=None)
    assert df.shape == (1, f)
    assert np.allclose(np.linalg.norm(df.values), 1.0, atol=1e-5)

# DINOv2 models
def test_image_features_frame_vits():
    run_backbone_test("vit-s", 384)

def test_image_features_frame_vitb():
    run_backbone_test("vit-b", 768)

def test_image_features_frame_vitl():
    run_backbone_test("vit-l", 1024, input_size=(64, 64))

def test_image_features_frame_vitg():
    run_backbone_test("vit-g", 1536, input_size=(64, 64))

# SigLIP models
def test_image_features_frame_siglip1():
    run_backbone_test("siglip1", 768)

def test_image_features_frame_siglip2():
    run_backbone_test("siglip2", 768)

def test_image_features_frame_siglip2_so400m_384():
    run_backbone_test("siglip2-so400m-384", 1152)

def test_image_features_frame_siglip2_so400m_512():
    run_backbone_test("siglip2-so400m-512", 1152)

def test_image_features_frame_siglip2_b16_256():
    run_backbone_test("siglip2-b16-256", 768)

def test_image_features_frame_siglip2_b16_512():
    run_backbone_test("siglip2-b16-512", 768)

def test_image_features_frame_siglip2_b32_256():
    run_backbone_test("siglip2-b32-256", 768)

# CLIP model

def test_image_features_frame_clip():
    run_backbone_test("clip", 512)

# DINOv3 models
def test_image_features_frame_dinov3_s():
    run_backbone_test("dinov3-s", 384)

def test_image_features_frame_dinov3_splus():
    run_backbone_test("dinov3-splus", 384)

def test_image_features_frame_dinov3_b():
    run_backbone_test("dinov3-b", 768)

def test_image_features_frame_dinov3_l():
    run_backbone_test("dinov3-l", 1024, input_size=(64, 64))

def test_image_features_frame_dinov3_hplus():
    run_backbone_test("dinov3-hplus", 1280, input_size=(64, 64))


def test_image_features_frame_dinov3_lsat():
    run_backbone_test("dinov3-l-sat", 1024)



# AM-RADIO models
def test_image_features_frame_radio_b():
    run_backbone_test("radio-b", 2304)

def test_image_features_frame_radio_l():
    run_backbone_test("radio-l", 3072)
    
def test_image_features_frame_radio_h():
    run_backbone_test("radio-h", 3840)

def test_image_features_frame_radio_g():
    run_backbone_test("radio-g", 4608)