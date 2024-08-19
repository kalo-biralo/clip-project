from preprocess import preprocess_image
import torch
import torch.nn as nn


def retrieve_text(model, image, texts, top_k=5):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = preprocessed_image.unsqueeze(0).to(model.device)

    model.eval()
    with torch.no_grad():
        image_embeddings = model.image_encoder(preprocessed_image)
        text_embeddings = model.text_encoder(texts, device=model.device)

        image_embeddings = model.image_projection_head(image_embeddings)
        text_embeddings = model.text_projection_head(text_embeddings)

        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)

    logits = image_embeddings @ text_embeddings.T
    top_matches = logits.softmax(dim=0)

    # Sort and pick top 5 indices
    _, top_k_matches = torch.topk(top_matches, k=top_k, dim=0)

    return logits[top_k_matches], top_k_matches


def retrieve_image(model, images, text, top_k=5):
    preprocessed_images = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        preprocessed_images.append(preprocessed_image)

    preprocessed_images = torch.stack(preprocessed_images).to(model.device)

    model.eval()
    with torch.no_grad():
        image_embeddings = model.image_encoder(preprocessed_images)
        text_embeddings = model.text_encoder(text, device=model.device)

        image_embeddings = model.image_projection_head(image_embeddings)
        text_embeddings = model.text_projection_head(text_embeddings)

        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)

    logits = image_embeddings @ text_embeddings.T
    top_matches = logits.softmax(dim=0)

    # Sort and pick top 5 indices
    _, top_k_matches = torch.topk(top_matches, k=top_k, dim=0)

    return logits[top_k_matches], top_k_matches
