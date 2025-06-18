from PIL import Image
from pathlib import Path
import numpy as np

img1 = Image.open("docs/resources/dinov2.jpg")
img2 = Image.open("docs/resources/siglip2.jpg")

height = img2.height
width = img2.width

img1 = img1.resize((width, height), Image.Resampling.LANCZOS)

width = img1.width + img2.width

combined_image = Image.new("RGB", (width, height))
combined_image.paste(img1, (0, 0))
combined_image.paste(img2, (img1.width, 0))
output_path = Path("docs/resources/combined_image.jpg")
combined_image.save(output_path)