import torch
from model import CLIP
from functions import retrieve_image, retrieve_text, similarity
import streamlit as st
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIP(device=device, pretrained=True)

# Streamlit UI
st.title("CLIP Model Image-Text Similarity")

option = st.selectbox(
    "Choose an option:", ("Calculate Similarity", "Retrieve Text", "Retrieve Image")
)


if option == "Calculate Similarity":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    text = st.text_input("Enter a description of the image:")

    if st.button("Submit"):
        if uploaded_file is not None and text:
            image = Image.open(uploaded_file)
            image = image.convert(
                "RGB"
            )  # Convert to RGB (if not already in RGB format)
            similarity_score = similarity(model, image, text)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write(f"Similarity: {similarity_score.item()}")
        else:
            st.write("Please upload an image and enter a description.")

elif option == "Retrieve Image":
    top_k = st.slider(
        "Select the number of top matches (k):", min_value=1, max_value=10, value=2
    )
    uploaded_files = st.file_uploader(
        "Choose images to retrieve similar texts:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    text_to_search = st.text_input("Enter text to search for similarity:")

    if st.button("Submit"):
        if uploaded_files and text_to_search:
            images = [Image.open(file).convert("RGB") for file in uploaded_files]
            logits, matches = retrieve_image(model, images, text_to_search, top_k)
            for idx, score in zip(matches, logits):
                st.image(
                    images[idx],
                    caption=f"Uploaded Image, Score: {score.item()}",
                    use_column_width=True,
                )
        else:
            st.write("Please upload images and enter text to search.")

elif option == "Retrieve Text":
    top_k = st.slider(
        "Select the number of top matches (k):", min_value=1, max_value=10, value=2
    )
    texts = st.text_input(
        "Enter text to search for similarity (comma-separated for multiple texts):"
    )
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.button("Submit"):
        if uploaded_file and texts:
            texts_list = [text.strip() for text in texts.split(",")]
            st.write("Texts to search:", texts_list)
            image = Image.open(uploaded_file)
            image = image.convert(
                "RGB"
            )  # Convert to RGB (if not already in RGB format)
            logits, matches = retrieve_text(model, image, texts_list, top_k)
            for idx, score in zip(matches, logits):
                st.write(f"Matching text: {texts_list[idx]}, Score: {score.item()}\n")
        else:
            st.write("Please upload an image and enter text to search.")
