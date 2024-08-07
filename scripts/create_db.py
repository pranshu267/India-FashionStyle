import faiss
import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel
import clip

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.load_state_dict(torch.load("clip_model.pth", map_location="cpu"))  # Adjust path as necessary
model.eval()


_, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)


def encode_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        image_feature = model.get_image_features(image).cpu().numpy()
    return image_feature


json_path = '/content/drive/MyDrive/fashion/train_data.json'
image_path = '/content/drive/MyDrive/fashion/images/train'

with open(json_path, 'r') as f:
    data = [json.loads(line) for line in f]


image_paths = []
image_embeddings = []

for item in data:
    img_path = os.path.join(image_path, item['image_path'].split('/')[-1])
    embedding = encode_image(img_path).flatten()
    image_paths.append(img_path)
    image_embeddings.append(embedding)


image_embeddings = np.array(image_embeddings).astype('float32')


index = faiss.IndexFlatL2(image_embeddings.shape[1])
index.add(image_embeddings)

faiss.write_index(index, "image_embeddings.index")

with open("image_paths.json", "w") as f:
    json.dump(image_paths, f)
