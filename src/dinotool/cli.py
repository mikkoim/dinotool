import torch
from dinotool import data
from dinotool.model import DinoFeatureExtractor, PCAModule, load_dino_model
from dinotool.utils import BatchHandler, frame_visualizer
import os
import uuid
from tqdm import tqdm
import warnings
import subprocess

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
        features = extractor(batch["img"])
        frame = data.FrameData(
            img=input["source"],
            features=features,
            pca=pca.transform(features, flattened=False)[0],
            frame_idx=0,
            flattened=False,
        )
        out_img = frame_visualizer(
            frame, output_size=input["input_size"], only_pca=args.only_pca
        )
        out_img.save(args.output)
        return

    batch_handler = BatchHandler(input["source"], extractor, pca)

    tmpdir = f"temp_frames-{str(uuid.uuid4())}"
    os.mkdir(tmpdir)
    video = input["source"]
    progbar = tqdm(total=len(video))
    for batch in input["data"]:
        batch_frames = batch_handler(batch)

        for frame in batch_frames:
            out_img = frame_visualizer(
                frame, output_size=input["input_size"], only_pca=args.only_pca
            )
            out_img.save(f"{tmpdir}/{frame.frame_idx:05d}.jpg")
            progbar.update(1)

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


if __name__ == "__main__":
    main()
