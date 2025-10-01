deactivate
rm -rf py311 py312 py313
uv venv py311 --python 3.11
uv venv py312 --python 3.12
uv venv py313 --python 3.13

make clear_tests
source py311/bin/activate
uv pip install .
uv pip install pytest
pytest

make clear_tests
deactivate
source py312/bin/activate
uv pip install .
uv pip install pytest
pytest


make clear_tests
deactivate
source py313/bin/activate
uv pip install .
uv pip install pytest
pytest

make clear_tests