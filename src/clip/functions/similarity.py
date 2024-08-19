from preprocess import preprocess_image
import torch
import torch.nn as nn


def similarity(model, image, text):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = preprocessed_image.unsqueeze(0).to(model.device)

    model.eval()
    with torch.no_grad():
        image_embeddings = model.image_encoder(preprocessed_image)
        text_embeddings = model.text_encoder(text, device=model.device)

        image_embeddings = model.image_projection_head(image_embeddings)
        text_embeddings = model.text_projection_head(text_embeddings)

        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)

    # logits = cosine_similarity(image_embeddings, text_embeddings)
    logits = image_embeddings @ text_embeddings.T

    return logits
