import pandas as pd
import numpy as np
import torch
from dinotool import data
from dinotool.model import DinoFeatureExtractor, OpenCLIPFeatureExtractor, PCAModule, load_model
from dinotool.utils import BatchHandler, frame_visualizer
import os
import uuid
from tqdm import tqdm
import subprocess
from pathlib import Path
import xarray as xr
from dataclasses import dataclass
from typing import Tuple, Dict, List, Literal
import shutil

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="🦕 DINOtool: Extract and visualize DINO features from images and videos."
    )
    parser.add_argument(
        "input", type=str, help="Path to an image, video file, or folder of images."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output visualization (image or video).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vits14_reg",
        help="DINO model to use (default: dinov2_vits14_reg).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        help="Resizes input to this size before passing it to the model (default: None).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1).",
    )
    parser.add_argument(
        "--only-pca",
        action="store_true",
        help="Only visualize PCA features (default: False).",
    )
    parser.add_argument(
        "--save-features",
        type=str,
        default=None,
        choices=["full", "flat", "frame"],
        help="Saves features to a netCDF file for images, and a zarr directory for videos (default: False).",
    )

    args = parser.parse_args()

    # Input validation
    if not os.path.exists(args.input):
        parser.error(f"Input path '{args.input}' does not exist.")
    if os.path.exists(args.output):
        parser.error(f"Output path '{args.output}' already exists.")

    return DinotoolConfig(
        input=args.input,
        output=args.output,
        model_name=args.model_name,
        input_size=tuple(args.input_size) if args.input_size else None,
        batch_size=args.batch_size,
        only_pca=args.only_pca,
        save_features=args.save_features,
    )


@dataclass
class DinotoolConfig:
    input: str
    output: str
    model_name: str = "dinov2_vits14_reg"
    input_size: Tuple[int, int] = None
    batch_size: int = 1
    only_pca: bool = False
    save_features: str = None


def save_batch_features(batch_frames: List[data.FrameData],
                        method: Literal["full", "flat"],
                        output: str):
    """
    Save features from a batch of frames to a file.
    Args:
        batch_frames (List[data.FrameData]): List of FrameData objects containing features.
        method (str): Method to save features, either "full" for netCDF or "flat" for parquet.
        output (str): Output file path without extension.
    Returns:
        None
    Side Effects:
        Saves features to a file in the specified format.
    """

    if method == "full":
        f_data = data.create_xarray_from_batch_frames(batch_frames)
        f_data.to_netcdf(f"{output}.nc")
    elif method == "flat":
        f_data = data.create_dataframe_from_batch_frames(batch_frames)
        f_data.to_parquet(f"{output}.parquet")


def combine_frame_features(method: Literal["full", "flat"],
                           tmpdir: str,
                           feature_out_name: str):
    """
    Combine features from temporary files into a single output file or directory.
    For "full" method, it combines netCDF files into a single zarr directory.
    For "flat" method, it moves parquet files into a specified directory into
    partitioned parquet files.
    Args:
        method (str): Method used to save features, either "full" for netCDF or "flat" for parquet.
        tmpdir (str): Temporary directory containing feature files.
        feature_out_name (str): Output file name without extension.
    Returns:
        None
    Side Effects:
        Saves features to a file or directory in the specified format.
    """
    if method == "full":
        nc_files = sorted(Path(tmpdir).glob("*.nc"))

        def process_one_path(path):
            with xr.open_dataset(path) as ds:
                ds.load()
                return ds

        xr_data = xr.concat([process_one_path(x) for x in nc_files], dim="frame_idx")
        xr_data.to_zarr(f"{feature_out_name}")
        print(f"Saving features to {feature_out_name}")
    elif method == "flat":
        Path(feature_out_name).mkdir(parents=True, exist_ok=True)
        parquet_files = sorted(Path(tmpdir).glob("*.parquet"))
        idx = 0
        for file in parquet_files:
            shutil.move(file, f"{feature_out_name}/part.{idx}.parquet")
            idx += 1
        print(f"Saved features to {feature_out_name}")

def handle_frame_level_features(input: torch.Tensor,
                                extractor: DinoFeatureExtractor,
                                output: str):
    """
    Handle frame-level/global feature extraction and saving.
    Args:
        input (torch.Tensor): Input video or tensor of frames.
        extractor (DinoFeatureExtractor): Feature extractor instance.
        output (str): Output file path without extension.
    """
    if isinstance(input, torch.Tensor):
        features = extractor(input, return_clstoken=True).cpu().numpy()
        np.savetxt(f"{output}.txt", features, delimiter=",")
        print(f"Saved features to {output}.txt")
        return

    print("Extracting frame-level features. This does not produce a video output.")
    tmpdir = f"temp_dinotool_frames-{str(uuid.uuid4())}"
    os.mkdir(tmpdir)

    try:
        progbar = tqdm(total=len(input))
        idx = 0
        for batch in input:
            features = extractor(batch["img"], return_clstoken=True).cpu().numpy()
            frame_idx = batch["frame_idx"].cpu().numpy()

            columns = [f"feature_{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, index=frame_idx, columns=columns)
            df.to_parquet(f"{tmpdir}/{idx:05d}.parquet")
            progbar.update(1)
            idx += 1
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        progbar.close()
    
    # collect all parquet files
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(Path(tmpdir).glob("*.parquet"))
    pd.concat(
        [pd.read_parquet(x) for x in parquet_files], axis=0
    ).to_parquet(f"{output}.parquet")

    # Clean up temporary files
    subprocess.run(["rm", "-r", f"{tmpdir}"])
    print(f"Saved features to {output}.parquet")

def load_extractor(model_name: str, 
                   model: torch.nn.Module = None,
                   input_size: Tuple[int, int] = None,
                   device: str = "cuda"):
    """
    Loads a feature extractor based on the model name.
    Args:
        model_name (str): Name of the model to load. Is used to determine the type of extractor.
        model (torch.nn.Module, optional): Preloaded model. Defaults to None.
        input_size (Tuple[int, int], optional): Input size for the model. Defaults to None.
        device (str, optional): Device to use for computation. Defaults to "cuda".
    Returns:
        DinoFeatureExtractor: Initialized feature extractor.
    """
    if model_name.startswith("hf-hub:timm"):
        extractor = OpenCLIPFeatureExtractor(
            model, input_size=input_size, device=device
        )
    else:
        extractor = DinoFeatureExtractor(model, input_size=input_size, device=device)
    return extractor

def main(args: DinotoolConfig):
    model = load_model(args.model_name)
    print(f"Using model: {args.model_name}")

    input = data.input_pipeline(
        args.model_name,
        args.input,
        patch_size=model.patch_size,
        batch_size=args.batch_size,
        resize_size=args.input_size,
    )
    print(f"Input size: {input['input_size']}")
    print(f"Feature map size: {input['feature_map_size']}")

    is_video = True
    if isinstance(input["data"], torch.Tensor):
        is_video = False
        batch = {}
        batch["img"] = input["data"]
    else:
        batch = next(iter(input["data"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    extractor = load_extractor(
        model_name=args.model_name,
        model=model,
        input_size=input["input_size"],
        device=device)

    if args.save_features == 'frame':
        handle_frame_level_features(input["data"],
                                    extractor=extractor,
                                    output=f"{Path(args.output).with_suffix('')}")
        return

    pca = PCAModule(n_components=3, feature_map_size=input["feature_map_size"])
    features = extractor(batch["img"])
    pca.fit(features)

    # Image input handling
    if not is_video:
        frame = data.FrameData(
            img=input["source"],
            features=extractor.reshape_features(features)[0],
            pca=pca.transform(features, flattened=False)[0],
            frame_idx=0,
            flattened=False,
        )
        out_img = frame_visualizer(
            frame, output_size=input["input_size"], only_pca=args.only_pca
        )
        out_img.save(args.output)
        print(f"Saved visualization to {Path(args.output)}")

        if args.save_features:
            save_batch_features(
                [frame],
                method=args.save_features,
                output=f"{Path(args.output).with_suffix('')}",
            )
            if args.save_features == "full":
                out_name = Path(args.output).with_suffix(".nc")
            else:
                out_name = Path(args.output).with_suffix(".parquet")
            print(f"Saved features to {out_name}")
        return

    # Video input handling
    if args.save_features:
        if args.save_features == "full":
            feature_out_name = Path(args.output).with_suffix(".zarr")
        else:
            feature_out_name = Path(args.output).with_suffix(".parquet")

    batch_handler = BatchHandler(input["source"], extractor, pca)

    tmpdir = f"temp_dinotool_frames-{str(uuid.uuid4())}"
    os.mkdir(tmpdir)
    video = input["source"]
    progbar = tqdm(total=len(video))
    try:
        for batch in input["data"]:
            batch_frames = batch_handler(batch)

            for frame in batch_frames:
                out_img = frame_visualizer(
                    frame, output_size=input["input_size"], only_pca=args.only_pca
                )
                out_img.save(f"{tmpdir}/{frame.frame_idx:05d}.jpg")
                progbar.update(1)

            if args.save_features:
                save_batch_features(
                    batch_frames,
                    method=args.save_features,
                    output=f"{tmpdir}/{batch_frames[0].frame_idx:05d}",
                )

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        progbar.close()

    if args.save_features:
        combine_frame_features(
            method=args.save_features, tmpdir=tmpdir, feature_out_name=feature_out_name
        )

    # Combine frames into a video using ffmpeg
    try:
        framerate = video.framerate
    except ValueError:
        framerate = 30

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(framerate),
            "-pattern_type",
            "glob",
            "-i",
            f"{tmpdir}/*.jpg",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            args.output,
        ]
    )

    # Clean up temporary files
    subprocess.run(["rm", "-r", f"{tmpdir}"])

    print(f"Saved visualization to {args.output}")
    if args.save_features:
        print(f"Saved features to {feature_out_name}")


def cli():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
