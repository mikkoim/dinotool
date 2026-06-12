# Basic tests
```bash
uv run pytest
```

# Longer tests
```bash
export HF_HUB_CACHE=/mnt/d/models
export TORCH_HOME=/mnt/d/models
uv run pytest -m slow -s 

# Failed 
export CUDA_VISIBLE_DEVICES=
export XFORMERS_DISABLED=1
uv run pytest -m slow -s --lf
```

# All python versions

```bash
bash test/test_all_python.sh
```

