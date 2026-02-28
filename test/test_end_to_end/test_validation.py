"""Tests for argument validation and error handling."""

import argparse
import os
import pytest
from pathlib import Path

from dinotool.cli import ArgumentValidator, DinotoolConfig, DinotoolProcessor


def test_validate_input_path_nonexistent():
    with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
        ArgumentValidator.validate_input_path("nonexistent/path.jpg")


def test_validate_input_path_exists():
    # Should not raise
    ArgumentValidator.validate_input_path("test/data/magpie.jpg")


def test_validate_output_path_exists_no_force():
    # Create a file first
    Path("test/outputs").mkdir(parents=True, exist_ok=True)
    Path("test/outputs/existing.txt").touch()
    with pytest.raises(argparse.ArgumentTypeError, match="already exists"):
        ArgumentValidator.validate_output_path("test/outputs/existing.txt", force=False)
    Path("test/outputs/existing.txt").unlink()


def test_validate_output_path_exists_with_force():
    Path("test/outputs").mkdir(parents=True, exist_ok=True)
    Path("test/outputs/existing.txt").touch()
    # Should not raise when force=True
    ArgumentValidator.validate_output_path("test/outputs/existing.txt", force=True)
    Path("test/outputs/existing.txt").unlink()


def test_validate_no_vis_without_save_features():
    with pytest.raises(argparse.ArgumentTypeError, match="--save-features"):
        ArgumentValidator.validate_vis_and_features(no_vis=True, save_features=None)


def test_validate_no_vis_with_save_features():
    # Should not raise
    ArgumentValidator.validate_vis_and_features(no_vis=True, save_features="frame")


def test_validate_imagedir_batch_without_no_vis():
    with pytest.raises(argparse.ArgumentTypeError, match="--no-vis"):
        ArgumentValidator.validate_input_type_and_vis_and_batch(
            input_type="image_directory",
            input_size=(224, 224),
            no_vis=False,
            input_path="test/data/imagefolder",
            batch_size=2,
        )


def test_validate_imagedir_batch_without_input_size():
    with pytest.raises(argparse.ArgumentTypeError, match="--input-size"):
        ArgumentValidator.validate_input_type_and_vis_and_batch(
            input_type="image_directory",
            input_size=None,
            no_vis=True,
            input_path="test/data/imagefolder",
            batch_size=2,
        )


def test_validate_imagedir_batch_valid():
    # Should not raise with batch_size=1
    ArgumentValidator.validate_input_type_and_vis_and_batch(
        input_type="image_directory",
        input_size=None,
        no_vis=False,
        input_path="test/data/imagefolder",
        batch_size=1,
    )
    # Should not raise with all requirements met
    ArgumentValidator.validate_input_type_and_vis_and_batch(
        input_type="image_directory",
        input_size=(224, 224),
        no_vis=True,
        input_path="test/data/imagefolder",
        batch_size=4,
    )


def test_validate_output_extension_invalid():
    with pytest.raises(argparse.ArgumentTypeError, match="valid extension"):
        ArgumentValidator.validate_output_extension(
            input_type="single_image",
            output_path="test/outputs/out.xyz",
            no_vis=False,
        )


def test_validate_output_extension_valid_with_no_vis():
    # Any extension is fine when no_vis is set
    ArgumentValidator.validate_output_extension(
        input_type="single_image",
        output_path="test/outputs/out_anything",
        no_vis=True,
    )


def test_validate_output_extension_imagedir_no_extension():
    # Image directories allow any output path (it becomes a directory)
    ArgumentValidator.validate_output_extension(
        input_type="image_directory",
        output_path="test/outputs/my_output_dir",
        no_vis=False,
    )


def test_force_overwrite():
    """Test that --force flag allows overwriting existing output."""
    Path("test/outputs").mkdir(parents=True, exist_ok=True)
    out_path = "test/outputs/force_test.jpg"

    # First run creates the file
    config = DinotoolConfig(
        input="test/data/magpie.jpg",
        output=out_path,
        input_size=(64, 64),
    )
    processor = DinotoolProcessor(config)
    processor.run()
    assert os.path.exists(out_path)
    first_mtime = os.path.getmtime(out_path)

    # Second run with same output should fail validation
    with pytest.raises(argparse.ArgumentTypeError, match="already exists"):
        ArgumentValidator.validate_output_path(out_path, force=False)

    # But validate_output_path with force=True should pass
    ArgumentValidator.validate_output_path(out_path, force=True)
