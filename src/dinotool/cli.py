import torch
from dinotool import data
from dinotool.model import DinoFeatureExtractor, PCAModule, load_dino_model
from dinotool.utils import BatchHandler, frame_visualizer
import os
import uuid
from tqdm import tqdm
import subprocess
from pathlib import Path
import xarray as xr

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
        action="store_true",
        help="Save features to a netcdf file using xarray (default: False).",
    )

    args = parser.parse_args()

    # Input validation
    if not os.path.exists(args.input):
        parser.error(f"Input path '{args.input}' does not exist.")
    if os.path.exists(args.output):
        parser.error(f"Output path '{args.output}' already exists.")

    return args


def main():
    args = parse_args()
    model = load_dino_model(args.model_name)

    input = data.input_pipeline(
        args.input,
        patch_size=model.patch_size,
        batch_size=args.batch_size,
        resize_size=args.input_size,
    )

    is_video = True
    if isinstance(input["data"], torch.Tensor):
        is_video = False

    extractor = DinoFeatureExtractor(model, input_size=input["input_size"])

    if is_video:
        batch = next(iter(input["data"]))
    else:
        batch = {}
        batch["img"] = input["data"]

    pca = PCAModule(n_components=3, feature_map_size=input["feature_map_size"])
    features = extractor(batch["img"])
    pca.fit(features)

    if not is_video:
        features_flat = extractor(batch["img"])
        frame = data.FrameData(
            img=input["source"],
            features=extractor.reshape_features(features_flat)[0],
            pca=pca.transform(features_flat, flattened=False)[0],
            frame_idx=0,
            flattened=False,
        )
        out_img = frame_visualizer(
            frame, output_size=input["input_size"], only_pca=args.only_pca
        )
        out_img.save(args.output)

        if args.save_features:
            f_data = data.create_xarray_from_batch_frames([frame])
            f_data.to_netcdf(args.output.replace(".jpg", ".nc"))
        return

    if args.save_features:
        feature_out_name = Path(args.output).with_suffix(".zarr")

    batch_handler = BatchHandler(input["source"], extractor, pca)

    tmpdir = f"temp_frames-{str(uuid.uuid4())}"
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
                f_data = data.create_xarray_from_batch_frames(batch_frames)
                f_data.to_netcdf(f"{tmpdir}/{frame.frame_idx:05d}.nc")

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        progbar.close()

    if args.save_features:
        nc_files = sorted(Path(tmpdir).glob("*.nc"))
        def process_one_path(path):
            with xr.open_dataset(path) as ds:
                ds.load()
                return ds
        xr_data = xr.concat([process_one_path(x) for x in nc_files], dim="frame_idx")
        xr_data.to_zarr(
            f"{feature_out_name}"
        )
        print(f"Saving features to {feature_out_name}")

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
    subprocess.run(["rm", "-r", f"{tmpdir}"])

    print(f"Saved visualization to {args.output}")
    if args.save_features:
        print(f"Saved features to {feature_out_name}")


if __name__ == "__main__":
    main()
