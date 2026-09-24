import pytest
import torch

from clip.model.clip import CLIP, ProjectionHead


def test_projection_head_keeps_batch_and_maps_to_projection_dim():
    head = ProjectionHead(embedding_dim=2048, projection_dim=768, dropout=0.2)
    assert head(torch.randn(4, 2048)).shape == (4, 768)


def test_load_weights_reports_a_missing_checkpoint(monkeypatch):
    # Bypass __init__ so no backbone weights are downloaded
    model = CLIP.__new__(CLIP)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    monkeypatch.delenv("CLIP_CHECKPOINT", raising=False)
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        model.load_weights("does/not/exist.pth")
