import torch

from ..preprocess import preprocess_image


def _top_k(scores, top_k):
    """Return (scores, indices) of the best matches, clamping k to what is available."""
    k = max(1, min(top_k, scores.numel()))
    values, indices = torch.topk(scores, k=k)
    return values, indices


def retrieve_text(model, image, texts, top_k=5):
    """
    Rank candidate `texts` against a single PIL `image`.

    Returns (scores, indices): cosine similarities of the top matches, sorted
    best first, and their positions in `texts`. `top_k` is clamped to len(texts).
    """
    image_tensor = preprocess_image(image).unsqueeze(0).to(model.device)

    model.eval()
    with torch.no_grad():
        image_embeddings = model.encode_image(image_tensor)  # (1, d)
        text_embeddings = model.encode_text(texts)  # (n_texts, d)

    scores = (image_embeddings @ text_embeddings.T).squeeze(0)  # (n_texts,)
    return _top_k(scores, top_k)


def retrieve_image(model, images, text, top_k=5):
    """
    Rank a list of PIL `images` against a single `text` query.

    Returns (scores, indices): cosine similarities of the top matches, sorted
    best first, and their positions in `images`. `top_k` is clamped to len(images).
    """
    image_tensors = torch.stack([preprocess_image(image) for image in images])
    image_tensors = image_tensors.to(model.device)

    model.eval()
    with torch.no_grad():
        image_embeddings = model.encode_image(image_tensors)  # (n_images, d)
        text_embeddings = model.encode_text(text)  # (1, d)

    scores = (image_embeddings @ text_embeddings.T).squeeze(1)  # (n_images,)
    return _top_k(scores, top_k)
