"""Integration tests for --resume / interrupted-run behavior."""

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dinotool.cli import DinotoolConfig, DinotoolProcessor, FeatureSaver


def _make_config(output, **kwargs):
    return DinotoolConfig(
        input="test/data/nasa.mp4",
        output=output,
        batch_size=8,
        no_vis=True,
        **kwargs,
    )


def _tmpdir(output):
    return str(Path(output).with_suffix("")) + ".dinotool_tmp"


def _run_interrupted(config, fail_after_batches: int):
    """Run DinotoolProcessor but raise KeyboardInterrupt after N save_batch_features calls."""
    call_count = 0
    original_save = FeatureSaver.save_batch_features  # already a plain function (staticmethod)

    def counting_save(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        original_save(*args, **kwargs)
        if call_count >= fail_after_batches:
            raise KeyboardInterrupt

    with patch.object(FeatureSaver, "save_batch_features", counting_save):
        with pytest.raises(KeyboardInterrupt):
            DinotoolProcessor(config).run()

    return call_count


@pytest.mark.slow
def test_resume_videofile_flat(tmp_path):
    """Resuming a flat-parquet run produces identical output to a clean run."""
    output = str(tmp_path / "nasa_resume.mp4")

    # Reference: full clean run
    ref_config = _make_config(output, save_features="flat")
    DinotoolProcessor(ref_config).run()
    reference = pd.read_parquet(str(tmp_path / "nasa_resume.parquet"))
    shutil.rmtree(str(tmp_path / "nasa_resume.parquet"))

    # Interrupted run: fail after 2 batches
    _run_interrupted(_make_config(output, save_features="flat"), fail_after_batches=2)

    tmpdir = _tmpdir(output)
    assert os.path.isdir(tmpdir), "tmpdir must survive the interrupt"
    saved = list(Path(tmpdir).glob("*.parquet"))
    assert len(saved) == 2, f"Expected 2 batch files, got {len(saved)}"

    # Resume
    resume_config = _make_config(output, save_features="flat", resume=True)
    DinotoolProcessor(resume_config).run()

    result = pd.read_parquet(str(tmp_path / "nasa_resume.parquet"))
    pd.testing.assert_frame_equal(reference.sort_index(), result.sort_index())


@pytest.mark.slow
def test_resume_videofile_all(tmp_path):
    """Resuming a --save-features all run produces identical local + global output."""
    output = str(tmp_path / "nasa_resume_all.mp4")

    # Reference
    ref_config = _make_config(output, save_features="all")
    DinotoolProcessor(ref_config).run()
    ref_local = pd.read_parquet(str(tmp_path / "nasa_resume_all.parquet"))
    ref_global = pd.read_parquet(str(tmp_path / "nasa_resume_all_frame.parquet"))
    shutil.rmtree(str(tmp_path / "nasa_resume_all.parquet"))
    shutil.rmtree(str(tmp_path / "nasa_resume_all_frame.parquet"))

    # Interrupted
    _run_interrupted(_make_config(output, save_features="all"), fail_after_batches=2)

    tmpdir = _tmpdir(output)
    assert os.path.isdir(tmpdir), "tmpdir must survive the interrupt"

    # Resume
    resume_config = _make_config(output, save_features="all", resume=True)
    DinotoolProcessor(resume_config).run()

    result_local = pd.read_parquet(str(tmp_path / "nasa_resume_all.parquet"))
    result_global = pd.read_parquet(str(tmp_path / "nasa_resume_all_frame.parquet"))
    pd.testing.assert_frame_equal(ref_local.sort_index(), result_local.sort_index())
    pd.testing.assert_frame_equal(ref_global.sort_index(), result_global.sort_index())


@pytest.mark.slow
def test_resume_tmpdir_deleted_on_success(tmp_path):
    """tmpdir is cleaned up after a successful (non-interrupted) run."""
    output = str(tmp_path / "nasa_clean.mp4")
    config = _make_config(output, save_features="flat")
    DinotoolProcessor(config).run()
    assert not os.path.exists(_tmpdir(output)), "tmpdir should be deleted after success"


@pytest.mark.slow
def test_resume_fresh_start_clears_stale_tmpdir(tmp_path):
    """Running without --resume clears any stale tmpdir and produces a clean output."""
    output = str(tmp_path / "nasa_fresh.mp4")

    # Create a stale tmpdir with a fake batch file
    tmpdir = _tmpdir(output)
    os.makedirs(tmpdir)
    Path(tmpdir, "00000.parquet").write_text("stale")

    config = _make_config(output, save_features="flat")
    DinotoolProcessor(config).run()

    result = pd.read_parquet(str(tmp_path / "nasa_fresh.parquet"))
    assert result.shape == (58140, 384)
    assert not os.path.exists(tmpdir), "tmpdir should be gone after successful run"
