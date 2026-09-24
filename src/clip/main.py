"""Streamlit demo: image-text similarity and retrieval with the CLIP model.

Run from the repository root:

    poetry run streamlit run src/clip/main.py
"""

import streamlit as st
import torch
from PIL import Image

from clip.functions import retrieve_image, retrieve_text, similarity
from clip.model import CLIP
from clip.model.clip import CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT

IMAGE_TYPES = ["jpg", "jpeg", "png"]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Build the model once per server process, not on every Streamlit rerun."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CLIP(device=device, pretrained=True)


st.title("CLIP Model Image-Text Similarity")

try:
    model = load_model()
except FileNotFoundError as error:
    st.error(
        f"{error}\n\nTrained weights are not bundled with the repository. Put the "
        f"checkpoint at `{DEFAULT_CHECKPOINT}` or set `{CHECKPOINT_ENV_VAR}` to its path."
    )
    st.stop()

option = st.selectbox(
    "Choose an option:", ("Calculate Similarity", "Retrieve Text", "Retrieve Image")
)


def open_image(file):
    return Image.open(file).convert("RGB")


if option == "Calculate Similarity":
    uploaded_file = st.file_uploader("Choose an image...", type=IMAGE_TYPES)
    text = st.text_input("Enter a description of the image:")

    if st.button("Submit"):
        if uploaded_file is not None and text:
            image = open_image(uploaded_file)
            score = similarity(model, image, text)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write(f"Similarity: {score.item():.4f}")
        else:
            st.write("Please upload an image and enter a description.")

elif option == "Retrieve Image":
    top_k = st.slider(
        "Select the number of top matches (k):", min_value=1, max_value=10, value=2
    )
    uploaded_files = st.file_uploader(
        "Choose images to search through:",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
    )
    text_to_search = st.text_input("Enter text to search for:")

    if st.button("Submit"):
        if uploaded_files and text_to_search:
            images = [open_image(file) for file in uploaded_files]
            scores, matches = retrieve_image(model, images, text_to_search, top_k)
            for idx, score in zip(matches.tolist(), scores.tolist()):
                st.image(
                    images[idx],
                    caption=f"Score: {score:.4f}",
                    use_container_width=True,
                )
        else:
            st.write("Please upload images and enter text to search.")

elif option == "Retrieve Text":
    top_k = st.slider(
        "Select the number of top matches (k):", min_value=1, max_value=10, value=2
    )
    texts = st.text_input(
        "Enter texts to compare against (comma-separated for multiple texts):"
    )
    uploaded_file = st.file_uploader("Choose an image...", type=IMAGE_TYPES)

    if st.button("Submit"):
        if uploaded_file and texts:
            texts_list = [text.strip() for text in texts.split(",") if text.strip()]
            st.write("Texts to search:", texts_list)
            scores, matches = retrieve_text(
                model, open_image(uploaded_file), texts_list, top_k
            )
            for idx, score in zip(matches.tolist(), scores.tolist()):
                st.write(f"Matching text: {texts_list[idx]}, Score: {score:.4f}")
        else:
            st.write("Please upload an image and enter text to search.")
