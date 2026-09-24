import torch

from ..preprocess import preprocess_image


def similarity(model, image, text):
    """Cosine similarity between one PIL image and one text; returns a (1, 1) tensor."""
    image_tensor = preprocess_image(image).unsqueeze(0).to(model.device)

    model.eval()
    with torch.no_grad():
        image_embeddings = model.encode_image(image_tensor)
        text_embeddings = model.encode_text(text)

    return image_embeddings @ text_embeddings.T
