deactivate
rm -rf py311 py312 py313
uv venv py311 --python 3.11
uv venv py312 --python 3.12
uv venv py313 --python 3.13

make clear_tests
source py311/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install .
uv pip install pytest
pytest

make clear_tests
deactivate
source py312/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install .
uv pip install pytest
pytest


make clear_tests
deactivate
source py313/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install .
uv pip install pytest
pytest

make clear_tests
rm -rf py311 py312 py313

# Slow tests
source .venv/bin/activate
export HF_HUB_CACHE=/mnt/d/models
export TORCH_HOME=/mnt/d/models
uv run pytest -m slow -s 

export CUDA_VISIBLE_DEVICES=
export XFORMERS_DISABLED=1
uv run pytest -m slow -s --lf