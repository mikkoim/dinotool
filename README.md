![PyPI](https://img.shields.io/pypi/v/dinotool)
![License](https://img.shields.io/github/license/mikkoim/dinotool)

# 🦕 DINOtool

**DINOtool** is a command-line tool for extracting visual features from images and videos using modern vision models like DINOv2, CLIP, SigLIP2, and OpenCLIP/timm compatible models.
It supports both **global (frame-level)** and **local (patch-level)** features, and can optionally visualize feature maps using PCA.

```bash
pip install dinotool
dinotool test.jpg -o out.jpg
```

## ✨ Features

- Works with:
  - 📷 Single images
  - 🎞️ Video files
  - 📁 Folders of images 

- 🧠 Supports multiple model backends:
  - DINOv2 (default)
  - SigLIP2, CLIP, and any timm/OpenCLIP model

- 💾 Outputs standard formats:
  - .parquet (flat/global features)
  - .zarr / .nc (spatial patch features)
  - .jpg / .mp4 with visualizations

- 🌈 Optional PCA-based side-by-side visualizations
- ⚡ Simple CLI with no coding required

## 👤 Who is DINOtool for?
DINOtool is designed for:

- Researchers exploring vision models or needing feature extraction for experiments

- Data scientists working with image/video datasets for tasks like clustering, retrieval, or classification

- Developers who want to use DINO, CLIP, or SigLIP2 features without writing model code

- Students and educators looking to visualize and understand patch-based ViT features

- Anyone who wants to preprocess media into standardized visual features for downstream ML tasks — without building a custom pipeline

## ✨Examples:
```bash
dinotool input.mp4 -o output.mp4
```
produces output:

[Video example](https://github.com/user-attachments/assets/0cc2e7ed-15b5-4f38-97f4-afee9b62e445)

DINOv2 accepts inputs of any size. The OpenCLIP/timm models resize the input. Here is an example of a 896x896 image:
```bash
dinotool test/data/bird1.jpg -o dinov2.jpg --model-name vit-b # Shortcut to dinov2_vitb14_reg
dinotool test/data/bird1.jpg -o siglip2.jpg --model-name siglip2 # Shortcut to hf-hub:timm/ViT-B-16-SigLIP2-512
```

produces outputs (DINOv2 / SigLIP2):

![DINO_SigLIP2](docs/resources/combined_image.jpg)

### Global features for image folders:

Processing image directories and extracting global or local features for each image is easy with DINOtool:

```bash
dinotool image_folder/ -o global_features --save-features 'frame'
```

produces a `global_features.parquet` file with global features:

| filename    | feature\_0 | feature\_1 | feature\_2 | ... | feature\_383 |
| -------------- | ---------- | ---------- | ---------- | --- | ------------ |
| `cat_001.jpg`  | 0.123      | -0.045     | 0.211      | ... | 0.009        |
| `dog_002.jpg`  | 0.097      | 0.033      | 0.187      | ... | -0.012       |
| `tree_003.jpg` | -0.056     | 0.140      | 0.092      | ... | 0.034        |
| `car_004.jpg`  | 0.301      | -0.202     | 0.144      | ... | -0.019       |

Similar files can be also produced for local patch features, for videos etc.

### More examples:

More example commands can be found in [test/test_cases.md](test/test_cases.md)

Example of reading output file formats is in [docs/reading_outputs.ipynb](docs/reading_outputs.ipynb)

Example of PCA feature visualization by first masking objects using the first PCA features, similar to DINOv2 demos is in [docs/masked_pca_demo.ipynb](docs/masked_pca_demo.ipynb):

![Masked_PCA](docs/resources/masked_pca.png)

## 📦 Installation

### Basic install (Linux/WSL2)

If you do not have ffmpeg installed:

```bash
sudo apt install ffmpeg
```

Install via pip:
```bash
pip install dinotool
```
You can check that dinotool is properly installed by testing it on an image:

```bash
dinotool test.jpg -o out.jpg
```

### `uv`

If you have `uv` installed, you can simply run DINOtool with
```bash
uv run --with dinotool dinotool test.jpg -o out.jpg
```
You still have to have `ffmpeg` installed. `uvx` does not work on linux due to `xformers` dependencies.

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

## 🚀 Basic usage

### 📷 Single images

Extract and visualize DINO features from an image:
```bash
dinotool input.jpg -o output.jpg
```
This produces a `.jpg` similar to the examples above.

For a easy-to-process Parquet file of the local features without visualization, run

```bash
dinotool input.jpg -o out_features --save-features 'flat' --no-vis
```

### 🎞️ Video:

Extract global features from a video using SigLIP2:
```bash
dinotool input.mp4 -o features --model-name siglip2 --save-features frame
```
This produces a `features.parquet` file with a row for each frame of the video.

### 📁 Folder of Images (or folders of video frames)

Process a folder of images with patch-level output:
```bash
dinotool images/ -o results --save-features full
```
This produces a folder `results` with visualization `.jpg` and a NetCDF file for each image separately.

If the images in the folder can be resized to a fixed size, you can use batch processing by setting a fixed resize size (`--input-size W H`) and `--no-vis`:
```bash
dinotool images/ -o results2 --save-features 'frame' --input-size 512 512 --batch-size 4 --no-vis
```
This produces a parquet file with global features for each image.

## 💾 Feature extraction options

Use `--save-features` to export features for downstream tasks.

| Mode     | Format                         | Output shape            |     Best for      |
|----------|--------------------------------|-------------------------|---------------------------|
| `full`   | `.nc` (image) / `.zarr` (video, batched image folders)| `(frames, height, width, feature)`|  Keeps spatial structure of patches.    |
| `flat`   | partitioned `.parquet`         | `(frames * height * width, feature)`|  Reliable long video processing. Faster patch-level analysis  |
| `frame`  | `.parquet`                     | `(frames, feature)`| One global feature vector per frame |

### `full` - Spatial local features
- Saves full patch feature maps from the ViT (one vector per image patch).
- Useful for reconstructing spatial attention maps or for downstream tasks like segmentation.
- Stored as netCDF for single images, `.zarr` for video sequences.
- `zarr` saving can be memory-intensive and might still fail for large videos.

```bash
dinotool input.mp4 -o output.mp4 --save-features full
```

### `flat` - Flattened local features
- Saves same vectors as above, but discards 2D spatial layout and saves output in `parquet` format.
- More reliable for longer videos.
- Useful for faster computations for statistics, patch-level similarity and clustering.
- For single image input saves a `.parquet` file with one row per patch.
- For video inputs saves a partitioned `.parquet` directory, with indices for frames and patches.

```bash
dinotool input.mp4 -o output.mp4 --save-features flat
```

### `frame` - Global features
- Saves one global feature vector per frame/image.
- Useful for temporal tasks, and creating vector databases.
- For single image input saves a `.txt` file with a single vector
- For image folder and video input saves a `.parquet` file with one row per frame/image.

```bash
# For a video
dinotool input.mp4 -o output.mp4 --save-features frame

# For an image
dinotool input.jpg -o output.jpg --save-features frame
```

The output is a side-by-side visualization with PCA of the patch-level features.

## 🧪 Additional Options

### `--model-name`

By default, the value passed to this argument is loaded from `facebookresearch/dinov2`, meaning that the possible DINOv2 models are:
- `dinov2_vits14`
- `dinov2_vitb14`
- `dinov2_vitl14`
- `dinov2_vitg14`

and their `reg` variants (recommended): i.e. `dinov2_vits14_reg`.

See the [DINOv2 github repo](https://github.com/facebookresearch/dinov2) for more information.

**OpenCLIP models:**

DINOtool now supports also ViT models that follow the OpenCLIP/timm model API for feature extraction. These models are for example the [SigLIP2 models in Huggingface hub](https://huggingface.co/collections/timm/siglip-2-67b8e72ba08b09dd97aecaf9). Additionally, [other models](https://huggingface.co/models?library=open_clip&sort=trending&search=timm%2F) in the Hub should also work, but have not been fully tested. These include SigLIP and CLIP models.

The OpenCLIP/timm model name has to be passed in the format `hf-hub:timm/<model name>`.

**Shortcuts**:
There are some predefined shortcuts for popular models. These can be passed to `--model-name`
```bash
# DINOv2
"vit-s": "dinov2_vits14_reg"
"vit-b": "dinov2_vitb14_reg"
"vit-l": "dinov2_vitl14_reg"
"vit-g": "dinov2_vitg14_reg"

# SigLIP2
"siglip2": "hf-hub:timm/ViT-B-16-SigLIP2-512"
"siglip2-so400m-384": "hf-hub:timm/ViT-SO400M-16-SigLIP2-384"
"siglip2-so400m-512": "hf-hub:timm/ViT-SO400M-16-SigLIP2-512"
"siglip2-b16-256": "hf-hub:timm/ViT-B-16-SigLIP2-256"
"siglip2-b16-512": "hf-hub:timm/ViT-B-16-SigLIP2-512"
"siglip2-b32-256": "hf-hub:timm/ViT-B-32-SigLIP2-256"
"siglip2-b32-512": "hf-hub:timm/ViT-B-32-SigLIP2-512"

# CLIP
"clip": "hf-hub:timm/vit_base_patch16_clip_224.openai"
```

## `--input-size`
Setting input size fixes the resolution for all inputs. This is useful for processing HD videos, and mandatory for batch processing of image folders.

```bash
# Processing a HD video faster:
dinotool input.mp4 -o output.mp4 --input-size 920 540 --batch-size 16
```

## `--batch-size`
For faster processing, set batch size as large as your GPU memory allows. Batch processing is possible for video files and directories of video frames (following naming where each imagename can be converted to an integer, like `00001.jpg`), where all inputs are assumed to be the same size.

```bash
dinotool input.mp4 -o output.mp4 --batch-size 16
```

For batch processing image folders, `--input-size` must be set. Visualization is also not possible.


## 🧑‍💻 Usage reference

```text
🦕 DINOtool: Extract and visualize ViT features from images and videos.

Usage:
  dinotool input_path -o output_path [options]

Arguments:
  input                   Path to image, video file, or folder of frames.
  -o, --output            Path for the output (required).

Options:
  -s, --save-features MODE    Save extracted features: full, flat, or frame
  -m, --model-name MODEL      Model to use (default: dinov2_vits14_reg)
  --input-size W H        Resize input before processing. Must be set for batch
                          processing of image folders
  -b, --batch-size N          Batch size for faster processing
  --only-pca              Only visualize PCA features.
  --no-vis                Only output features with no visualization.
                          --save features must be set.
```
