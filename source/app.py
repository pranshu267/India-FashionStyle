import streamlit as st
from PIL import Image
import numpy as np
import torch
import json
import faiss
from transformers import CLIPModel
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load("clip_model.pth", map_location=device))  
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_preprocess():
    _, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return preprocess

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("image_embeddings.index")
    with open("image_paths.json", "r") as f:
        image_paths = json.load(f)
    return index, image_paths


def get_top_3_images(text_input, model, index, image_paths):

    text_inputs = clip.tokenize([text_input]).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(text_inputs).cpu().numpy()

    _, top_3_indices = index.search(text_features, 3)

    top_3_images = [image_paths[i] for i in top_3_indices[0]]
    return top_3_images

def main():
    st.title("Indo Fashion CLIP Model")

    model = load_model()

    index, image_paths = load_faiss_index()

    text_input = st.text_input("Enter a Indo clothing description:")

    if st.button("Find Indo Clothing Recommendations"):
        if text_input:

            top_3_images = get_top_3_images(text_input, model, index, image_paths)
            st.write("Top 3 Recommended Clothes:")

            for i, img_path in enumerate(top_3_images):
                image = Image.open(img_path)
                st.image(image, caption=f"Match {i + 1}")

if __name__ == "__main__":
    main()
