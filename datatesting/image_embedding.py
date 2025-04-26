import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, requests, io
from pathlib import Path

CSV_IN  = Path("media.csv")
CSV_OUT = Path("places_with_image_embeddings.csv")

df = pd.read_csv(CSV_IN)

# Load CLIP
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def get_image_emb(url: str):
    resp = requests.get(url, timeout=10); resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb[0].cpu().numpy()

print("About to start embedding", len(df), "images")
df["img_emb"] = df["media_url"].apply(get_image_emb)
print("Finished embeddings, now writing CSV")

# Expand and save
emb_df = pd.DataFrame(
    df["img_emb"].tolist(),
    columns=[f"img_emb_{i}" for i in range(df["img_emb"].iloc[0].shape[0])]
)
out = pd.concat([df[["place_id","media_url"]], emb_df], axis=1)
out.to_csv(CSV_OUT, index=False)
print(f"Wrote {CSV_OUT}")
