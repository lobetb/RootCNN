import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2

import argparse

# ---------- UNet Model ----------
class UNetBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.pool = torch.nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(256, 512)
        self.up3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.final = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


# ---------- Heatmap Inference ----------
def predict_full_image(model, image_np, patch_size=512, stride=256, device='cuda'):
    model.eval()
    H, W = image_np.shape[:2]
    full_heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in tqdm(range(0, H - patch_size + 1, stride), desc="Predicting"):
            for x in range(0, W - patch_size + 1, stride):
                patch = image_np[y:y+patch_size, x:x+patch_size]
                if patch.ndim == 2:
                    patch = np.stack([patch]*3, axis=-1)
                patch = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                patch = patch.to(device)

                with torch.amp.autocast(device_type='cuda'):
                    out = model(patch)
                    heat = torch.sigmoid(out).squeeze().cpu().numpy()

                full_heatmap[y:y+patch_size, x:x+patch_size] += heat
                counts[y:y+patch_size, x:x+patch_size] += 1

    full_heatmap = np.divide(full_heatmap, counts, out=np.zeros_like(full_heatmap), where=counts > 0)
    return full_heatmap


# ---------- Visualization ----------
def overlay_heatmap(image_np, heatmap, alpha=0.5):
    heatmap_norm = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return blended

def batch_generate_heatmaps(input_dir, model_path, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in Path(input_dir).iterdir() if f.suffix.lower() in (".jpg", ".png", ".tif")]

    for img_path in img_files:
        print(f"Processing {img_path}")
        image = np.array(Image.open(img_path).convert("RGB"))
        heatmap = predict_full_image(model, image, patch_size=512, stride=256, device=device)
        vis = overlay_heatmap(image, heatmap)
        out_path = Path(output_dir) / (img_path.stem + "_heatmap.jpg")
        cv2.imwrite(str(out_path), vis[:, :, ::-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input images folder")
    parser.add_argument("--model", default="outputs/model.pth", help="Model path")
    parser.add_argument("--output_dir", required=True, help="Save results folder")
    args = parser.parse_args()

    batch_generate_heatmaps(args.input_dir, args.model, args.output_dir)
