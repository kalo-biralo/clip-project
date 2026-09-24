"""Tests for preprocessing and retrieval logic.

A stub model is used so the tests run offline and without the trained weights.
"""

import pytest
import torch
from PIL import Image

from clip.functions import retrieve_image, retrieve_text, similarity
from clip.preprocess import preprocess_image

COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
TEXT_VECTORS = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
}


class StubModel:
    """Embeds an image as its per-channel mean and a text as a fixed colour vector."""

    device = torch.device("cpu")

    def eval(self):
        return self

    def encode_image(self, images):
        return torch.nn.functional.normalize(images.mean(dim=(2, 3)), dim=-1)

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return torch.tensor([TEXT_VECTORS[text] for text in texts])


def solid(name):
    return Image.new("RGB", (64, 48), COLOURS[name])


def test_preprocess_resizes_and_returns_chw_tensor():
    tensor = preprocess_image(solid("red"))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_similarity_is_higher_for_the_matching_text():
    model = StubModel()
    match = similarity(model, solid("red"), "red")
    mismatch = similarity(model, solid("red"), "blue")
    assert match.shape == (1, 1)
    assert match.item() > mismatch.item()


def test_retrieve_text_ranks_the_matching_text_first():
    scores, indices = retrieve_text(
        StubModel(), solid("green"), ["red", "green", "blue"], top_k=2
    )
    assert indices[0].item() == 1
    assert len(scores) == len(indices) == 2
    assert scores[0] >= scores[1]


def test_retrieve_image_ranks_the_matching_image_first():
    images = [solid("red"), solid("green"), solid("blue")]
    scores, indices = retrieve_image(StubModel(), images, "blue", top_k=3)
    assert indices[0].item() == 2
    assert list(scores) == sorted(scores.tolist(), reverse=True)


@pytest.mark.parametrize("top_k", [5, 10])
def test_top_k_is_clamped_to_the_number_of_candidates(top_k):
    scores, indices = retrieve_text(
        StubModel(), solid("red"), ["red", "blue"], top_k=top_k
    )
    assert len(indices) == 2

    scores, indices = retrieve_image(StubModel(), [solid("red")], "red", top_k=top_k)
    assert len(indices) == 1
