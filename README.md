# 🦕 DINOtool

**DINOtool** is a simple Python package that makes it easy to extract and visualize features from images and videos using [DINOv2](https://dinov2.metademolab.com/) models.
**DINOtool** helps you generate frame and patch-level embeddings with a single command.

## ✨ Features

- 📷 Extract DINO features from:
  - Single images
  - Video files (`.mp4`, `.avi`, etc.)
  - Folders containing image sequences
- 🌈 Automatically generates PCA visualizations of the features
- 🧠 Visuals include side-by-side view of the original frame and the feature map
- Saves features for downstream tasks
- ⚡ Command-line interface for easy, no-code operation

[Video example](https://github.com/user-attachments/assets/0cc2e7ed-15b5-4f38-97f4-afee9b62e445)

## 📦 Installation

### Basic install (Linux/WSL2)

Install via pip:

```bash
pip install dinotool
```
You'll also need to have ffmpeg installed:

```bash
sudo apt install ffmpeg
```
You can check that dinotool is properly installed by testing it on an image:

```bash
dinotool test.jpg -o out.jpg
```

### 🐍 Conda Environment (Recommended)
If you want an isolated setup, especially useful for managing `ffmpeg` and dependencies:

Install [Miniforge](https://conda-forge.org/download/).

```bash
conda create -n dinotool python=3.12
conda activate dinotool
conda install -c conda-forge ffmpeg
pip install dinotool
```

### Windows notes:
- Windows is supported only for CPU usage. If you want GPU support on Windows, we recommend using WSL2 + Ubuntu.
- The conda method above is recommended for Windows CPU setups.

## 🚀 Quickstart

📸 Image:
```bash
dinotool input.jpg -o output.jpg
```

🎞️ Video
```bash
dinotool input.mp4 -o output.mp4
```

📁 Folder of Images (treated as video frames)
```bash
dinotool path/to/folder/ -o output.mp4
```

The output is a side-by-side visualization with PCA of the patch-level features.

## 🧪 Advanced Options

| Flag                | Description                                                           |
|---------------------|------------------------------------------------------------------------|
| `--model-name`      | Use a different DINO model (default: `dinov2_vits14_reg`)             |
| `--input-size W H`  | Resize input before inference                                          |
| `--batch-size`      | Batch size for processing (default: 1)                                 |
| `--only-pca`        | Output *only* the PCA map, without side-by-side                        |
| `--save-features`   | Save extracted features: `full`, `flat`, or `frame`                   |
| `-o, --output`      | Output path (required)                                                 |

## Tips:
Increase `--batch-size` to the largest value your memory supports for faster processing. 

```bash
dinotool input.mp4 -o output.mp4 --batch-size 16
```

For large videos, reduce the input size with `--input-size`

```bash
# Processing a HD video faster:
dinotool input.mp4 -o output.mp4 --input-size 920 540 --batch-size 16
```


## 💾 Feature extraction options

Use `--save-features` to export DINO features for downstream tasks.

| Mode     | Format                         | Output shape            |     Best for      |
|----------|--------------------------------|-------------------------|---------------------------|
| `full`   | `.nc` (image) / `.zarr` (video)| `(frames, height, width, feature)`|  Keeps spatial structure of patches.    |
| `flat`   | partitioned `.parquet`         | `(frames * height * weight, feature)`|  Reliable long video processing. Faster patch-level analysis  |
| `frame`  | `.parquet`                     | `(frames, feature)`| One feature vector per frame (global content representation) |

### `full` - Spatial patch features
- Saves full patch feature maps from the ViT (one vector per image patch).
- Useful for reconstructing spatial attention maps or for downstream tasks like segmentation.
- Stored as netCDF for single images, `.zarr` for video sequences.
- `zarr` saving can be memory-intensive and might still fail for large videos.

```bash
dinotool input.mp4 -o output.mp4 --save-features full
```

### `flat` - Flattened patch features
- Saves same vectors as above, but discards 2D spatial layout and saves output in `parquet` format.
- More reliable for longer videos.
- Useful for faster computations for statistics, patch-level similarity and clustering.

```bash
dinotool input.mp4 -o output.mp4 --save-features flat
```

### `frame` - Frame-level features
- Saves one vector per frame using the `[CLS]` token from DINO.
- Useful for temporal tasks, video summarization and classification.
- For image input saves a `.txt` file with a single vector
- For video input saves a `.parquet` file with one row per frame.

```bash
# For a video
dinotool input.mp4 -o output.mp4 --save-features frame

# For an image
dinotool input.jpg -o output.jpg --save-features frame
```

## 🧑‍💻 Usage reference

```text
🦕 DINOtool: Extract and visualize DINO features from images and videos.

Usage:
  dinotool input_path -o output_path [options]

Arguments:
  input                   Path to image, video file, or folder of frames.
  -o, --output            Path for the output (required).

Options:
  --model-name MODEL      DINO model to use (default: dinov2_vits14_reg)
  --input-size W H        Resize input before processing
  --batch-size N          Batch size for inference
  --only-pca              Only visualize PCA features
  --save-features MODE    Save extracted features: full, flat, or frame
```
