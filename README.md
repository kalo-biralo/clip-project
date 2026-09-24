# CLIP-style image–text retrieval

A compact, CPU-friendly implementation of the CLIP idea: embed images and text into one shared space with a contrastive objective, then use cosine similarity for zero-shot matching and retrieval. It includes the model, retrieval helpers, tests and a Streamlit demo.

> **Status.** The model code, retrieval functions and demo are here. The trained checkpoint and the training script are **not** part of this repository, so the demo needs a checkpoint supplied by you (see [Weights](#weights)).

## How it works

```
image ──► ResNet-50 (frozen) ──► projection head ─┐
                                                   ├─► 768-d unit vectors ─► cosine similarity
text  ──► DistilBERT  (frozen) ──► projection head ─┘
```

- **Backbones are pretrained and frozen.** ResNet-50 (ImageNet weights) encodes images and DistilBERT (`distilbert-base-uncased`, `[CLS]` token) encodes text. This is a deliberate trade-off: only the two projection heads are trained (about 3.3M trainable parameters against 89.9M frozen), which is far cheaper than training both encoders from scratch.
- **Projection heads** are `Linear → GELU → Linear → Dropout`, with a residual connection from the first linear layer and a final `LayerNorm`, mapping both modalities to 768 dimensions.
- **Loss** is the symmetric contrastive (InfoNCE) loss over a batch of matching pairs, with a learnable temperature initialised to `log(1/0.07)` and clamped to a scale of at most 100, as in the original paper.

See `CLIP.forward` in [`src/clip/model/clip.py`](src/clip/model/clip.py).

## What you can do with it

| Function | Input | Output |
| --- | --- | --- |
| `similarity(model, image, text)` | one image, one text | cosine similarity |
| `retrieve_text(model, image, texts, top_k)` | one image, many candidate texts | best-matching texts (zero-shot classification is this with class names as prompts) |
| `retrieve_image(model, images, text, top_k)` | many images, one text | best-matching images (text-to-image search) |

## Project layout

```
src/clip/
  main.py              Streamlit demo
  model/clip.py        ImageEncoder, TextEncoder, ProjectionHead, CLIP
  functions/           similarity and retrieval helpers
  preprocess/          image preprocessing (resize to 224x224, ImageNet normalisation)
tests/                 offline unit tests (no weights or downloads needed)
Dockerfile             container for the demo
```

## Getting started

Requires Python 3.10–3.12 and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

On Linux and Windows (x86-64) this installs the CPU-only PyTorch build; on macOS and ARM machines it uses the regular PyPI wheels, which are already CPU-only.

### Weights

The model needs a checkpoint containing a `model_state_dict` for the projection heads and temperature. The path is resolved in this order:

1. the `checkpoint_path` argument to `CLIP(...)` or `CLIP.load_weights(...)`
2. the `CLIP_CHECKPOINT` environment variable
3. `weights/best_checkpoint.pth` relative to the working directory

Checkpoints are loaded with `torch.load(..., weights_only=True)`. The ResNet-50 and DistilBERT backbones are downloaded automatically on first use.

### Python API

```python
import torch
from PIL import Image

from clip.functions import retrieve_text
from clip.model import CLIP

model = CLIP(device=torch.device("cpu"), pretrained=True)

image = Image.open("dog.jpg").convert("RGB")
scores, indices = retrieve_text(
    model, image, ["a photo of a dog", "a photo of a cat", "a photo of a car"], top_k=2
)
```

### Demo app

```bash
poetry run streamlit run src/clip/main.py
```

### Docker

```bash
docker build -t clip-demo .
docker run -p 8501:8501 -v "$(pwd)/weights:/app/weights" clip-demo
```

Then open http://localhost:8501. Weights are mounted rather than baked into the image.

## Development

```bash
poetry run pytest              # unit tests (offline)
poetry run black src tests     # formatting
poetry run pre-commit install  # optional git hooks
```

CI runs formatting and tests on every push and pull request.

## Not yet included

- The training script and the dataset it used
- Evaluation results (for example Recall@K on a held-out set)
- A published checkpoint

## References

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021)
- [OpenAI: CLIP — Connecting Text and Images](https://openai.com/research/clip)
