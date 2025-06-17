
```bash
# Single image input:
uv run dinotool test/data/bird1.jpg -o out.jpg --save-features 'flat'

# Video input
uv run dinotool test/data/nasa.mp4 -o nasa.mp4 --batch-size 4
# -> Outputs nasa.mp4

# "I want to use SigLIP2 instead"
uv run dinotool test/data/nasa.mp4 -o nasa-siglip2.mp4 --batch-size 4 --model-name siglip2
# -> Outputs nasa-siglip2.mp4
# CLIP/SigLIP inputs are resized based on the preprocessing pipeline of the model.

# "I want to save features but no need for visualization"
uv run dinotool test/data/nasa.mp4 -o nasa --save-features 'flat' --no-vis
# -> Produces partitioned `nasa.parquet` -directory.

# "I just want the local features of this image in a easily readable parquet format"
uv run dinotool test/data/bird1.jpg -o bird_features --save-features 'flat' --no-vis
# -> Produces bird_features.parquet

# I have a lot of images that are different sizes, and I want their local features in a format that preserves the locality
# I also want to save the PCA outputs for visual inspection
uv run dinotool test/data/imagefolder -o my_imagefolder --save-features 'full'

# Want to get global feature vectors with SigLIP2 - no need for visualization
uv run dinotool test/data/imagefolder -o my_imagefolder --save-features 'frame'

# I have a folder of images but they can be all resized to the same size for faster batch processing
uv run dinotool test/data/imagefolder -o my_imagefolder --save-features 'full' --input-size 960 540 --batch-size 4 --no-vis
```