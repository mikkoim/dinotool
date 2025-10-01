![PyPI](https://img.shields.io/pypi/v/dinotool)
![License](https://img.shields.io/github/license/mikkoim/dinotool)

# 🦕 DINOtool

**DINOtool** is a command-line tool for extracting visual features using state-of-the-art foundation models like DINOv3, DINOv2, CLIP, SigLIP2 and AM-RADIO.
It supports the extraction **global (frame-level)** and **local (patch-level)** features from images, videos and image directories, and can optionally visualize feature maps using PCA.

### Use from the command line
```bash
pip install dinotool
dinotool test.jpg -o out.jpg
```

### Use as a package:
```python
from dinotool import DinoToolModel
from PIL import Image
import matplotlib.pyplot as plt

model = DinoToolModel("dinov3-s") # Unified API for multiple model backends
img = Image.open("../test/data/bird1.jpg")
transform = model.get_transform(img.size) # Loads model-specific transforms
img_tensor = transform.transform(img).unsqueeze(0)

local_features = model(img_tensor)
local_features.tensor.shape # torch.Size([1, 56, 56, 384])

global_features = model(img_tensor, features="frame")
global_features.tensor.shape # torch.Size([1, 384])

plt.imshow(model.pca(features)) # PCA visualization
```

# Features

- Works with:
  - 📷 Single images
  - 🎞️ Video files
  - 📁 Folders of images of various sizes and extensions

- 💾 Outputs standard formats for downstream processing:
  - .parquet (global features / local features in a list)
  - .zarr / .nc (local features in 2D)

- 🌈 PCA-based visualizations for images and video

# Supported models

- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3) (Gated model - requires logging in to Hugginface Hub and applying access at [each model page](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009))
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [CLIP](https://github.com/openai/CLIP)
- [AM-RADIO](https://github.com/NVlabs/RADIO)

List available models and their shortcuts with `dinotool --models`.
# Examples:
```bash
dinotool input.mp4 -o output.mp4
```
produces output:

[Video example](https://github.com/user-attachments/assets/0cc2e7ed-15b5-4f38-97f4-afee9b62e445)

Changing the model with `--model-name/-m`:
```bash
dinotool bird.jpg out.jpg --model-name dinov3-s
```
uses a different foundation model to extract the features:

![DINO_SigLIP2](docs/resources/model_comparison.png)

### Global features for image folders:

Feature vectors can be saved with `--save-features`.
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

More example commands for different situations can be found in [test/test_cases.md](test/test_cases.md)

Example of reading output file formats is in [docs/reading_outputs.ipynb](docs/reading_outputs.ipynb)

Example of using DINOtool as a package is in [docs/api_example.ipynb](docs/api_example.ipynb)

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
- Windows is supported only for CPU usage. If you want GPU support on Windows, we recommend using WSL2 + Ubuntu, or managing the environment yourself by first installing a GPU torch version and then dinotool.
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

List available models and their shortcuts with `dinotool --models`.

By default, the value passed to `--model-name` argument is loaded from `facebookresearch/dinov2`, meaning that the possible DINOv2 models are:
- `dinov2_vits14`
- `dinov2_vitb14`
- `dinov2_vitl14`
- `dinov2_vitg14`

and their `reg` variants (recommended): i.e. `dinov2_vits14_reg`.

See the [DINOv2 github repo](https://github.com/facebookresearch/dinov2) for more information.

**DINOv3 models:**

Model names with prefix `facebook/dinov3/<model name>` are downloaded from the DINOv3 respository in Huggingface Hub. See a list of available models [here](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009).

[!IMPORTANT]
The DINOv3 models are gated models and need authorized access. You have to apply for access on the model page when logged in to Huggingface, and log in on the HF CLI: `hf auth login`.

**AM-RADIO models:**

Model names with prefix `NVlabs/RADIO/` are downloaded from the RADIO family of models. See all available models [here](https://github.com/NVlabs/RADIO)

**OpenCLIP models:**

DINOtool supports also ViT models that follow the OpenCLIP/timm model API for feature extraction. These models are for example the [SigLIP2 models in Huggingface hub](https://huggingface.co/collections/timm/siglip-2-67b8e72ba08b09dd97aecaf9). Additionally, [other models](https://huggingface.co/models?library=open_clip&sort=trending&search=timm%2F) in the Hub should also work, but have not been fully tested. These include SigLIP and CLIP models.

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
  -f, --force             Force overwrite output file if it exists.
  --models                List available models and their shortcuts.
  --version               Show the version of DINOtool.
```
