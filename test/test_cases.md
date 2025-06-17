
```bash
# Single image input:
uv run dinotool test/data/bird1.jpg -o out.jpg

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

# Image folder input
uv run dinotool test/data/drone_images -o drone
```