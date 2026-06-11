# Adding New Models to DINOtool

New model families require changes in four places, all wired together by **name-prefix routing** — the same `model_name` string is checked with `.startswith()` in each location.

---

## Checklist

### 1. Load the model — `src/dinotool/model.py`, `load_model()`

Add a branch that returns the raw `nn.Module` and sets `model.patch_size`:

```python
elif model_name.startswith("myorg/mymodel"):
    model = load_from_wherever(model_name)
    model.patch_size = model.config.patch_size  # or however the model exposes it
```

`patch_size` is read later by the extractor and the transform factory, so it must be set here.

---

### 2. Write an extractor class — `src/dinotool/model.py`

Subclass `nn.Module`. The only method that matters is `forward()`, which must accept these four arguments and return a `LocalFeatures` object (or a plain tensor when `return_clstoken=True`):

```python
class MyModelFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, device: str = "cuda"):
        super().__init__()
        self.model = model.eval().to(device)
        self.device = device
        self.patch_size = model.patch_size

    def forward(
        self,
        batch: torch.Tensor,       # (b, c, h, w), already preprocessed
        flattened: bool = True,
        normalized: bool = True,
        return_clstoken: bool = False,
    ):
        b, c, h, w = batch.shape
        dims = calculate_dino_dimensions((w, h), self.patch_size)
        h_featmap, w_featmap = dims["h_featmap"], dims["w_featmap"]

        with torch.no_grad():
            batch = batch.to(self.device)
            # --- call the model ---
            output = self.model(batch)

        if return_clstoken:
            cls = output.cls_token               # shape (b, f)
            if normalized:
                cls = torch.nn.functional.normalize(cls, dim=-1)
            return cls

        # patch tokens — shape must end up as (b, h_featmap, w_featmap, f)
        # or (b, h_featmap*w_featmap, f) — LocalFeatures handles both
        patch_tokens = output.patch_tokens      # example: (b, h*w, f)

        features = LocalFeatures(
            patch_tokens, is_flattened=True, h=h_featmap, w=w_featmap
        )
        if normalized:
            features = features.normalize()
        return features.flat() if flattened else features.full()
```

**`LocalFeatures` shape rules:**
- Pass `is_flattened=True` when the tensor is `(b, h*w, f)`
- Pass `is_flattened=False` when the tensor is `(b, h, w, f)`
- Always pass `h=h_featmap, w=w_featmap` so `.full()` / `.flat()` can convert between layouts

---

### 3. Add a transform — `src/dinotool/data.py`, `TransformFactory`

**A. Detect the model family in `__init__`:**

```python
elif model_name.startswith("myorg/mymodel"):
    self.model_type = "mymodel"
```

**B. Add a `get_mymodel_transform()` method:**

```python
def get_mymodel_transform(self, input_size: Tuple[int, int]):
    if input_size in self._transform_cache:
        return self._transform_cache[input_size]

    dims = calculate_dino_dimensions(input_size, patch_size=self.patch_size)
    model_input_size = (dims["w"], dims["h"])
    feature_map_size = (dims["w_featmap"], dims["h_featmap"])

    transform = transforms.Compose([
        transforms.Resize((model_input_size[1], model_input_size[0])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[...], std=[...]),  # model-specific stats
    ])

    result = DINOTransform(           # reuse any existing dataclass, or add a new one
        transform=transform,
        resize_size=model_input_size,
        feature_map_size=feature_map_size,
    )
    self._transform_cache[input_size] = result
    return result
```

**C. Route in `get_transform()`:**

```python
elif self.model_type == "mymodel":
    return self.get_mymodel_transform(input_size)
```

**Fixed vs. variable input size:** OpenCLIP models have a fixed size baked into their transform, so `get_openclip_transform()` ignores `input_size` and caches on `self.transform`. All other model families accept variable sizes — cache by `input_size` in `_transform_cache`.

If your transform is variable-size, create a dataclass for it (like `DINOTransform` / `RADIOTransform`) so the rest of the pipeline can read `resize_size` and `feature_map_size` off it consistently.

---

### 4. Wire up the CLI — `src/dinotool/cli.py`

**A. Add shortcuts in `MODEL_SHORTCUTS` (line ~34):**

```python
"mymodel-b": "myorg/mymodel-base",
"mymodel-l": "myorg/mymodel-large",
```

**B. Route to your extractor in `ExtractorFactory.create_extractor()` (line ~395):**

```python
elif model_name.startswith("myorg/mymodel"):
    return MyModelFeatureExtractor(model, device=device)
```

Also add the import at the top of `cli.py`:

```python
from dinotool.model import (
    ...
    MyModelFeatureExtractor,
)
```

---

## Verification

```bash
# single image
uv run dinotool test/data/bird1.jpg -o out.jpg -m mymodel-b

# save patch features
uv run dinotool test/data/bird1.jpg -o out.jpg -m mymodel-b --save-features flat

# global CLS token
uv run dinotool test/data/bird1.jpg -o out.jpg -m mymodel-b --save-features frame

# run against existing tests (will catch transform/extractor wiring issues)
uv run pytest
```
