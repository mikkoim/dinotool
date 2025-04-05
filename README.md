# 🦕 DINOtool

**DINOtool** is a simple Python package that extracts and visualizes features using the [DINO](https://dinov2.metademolab.com/) self-supervised vision transformer model. **DINOtool** helps you generate feature maps and colorful PCA visualizations of feature embeddings with just a single command.

## ✨ Features

- 📷 Extract DINO features from:
  - Single images
  - Video files (`.mp4`, `.avi`, etc.)
  - Folders containing image sequences
- 🌈 Automatically generates PCA visualizations of the features
- 🧠 Visuals include side-by-side view of the original frame and the feature map
- ⚡ Command-line interface for easy, no-code operation

## 📦 Installation

Install via pip:

```bash
pip install dinotool
```

You have to have `ffmpeg` installed.

# 🚀 Quickstart

From an Image:

```bash
dinotool path/to/image.jpg
```

From a video file:
```bash
dinotool path/to/video.mp4
```

From a folder of images:
```bash
dinotool path/to/folder/
```
