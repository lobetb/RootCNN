import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path

def make_heatmap(coords, size, sigma=3):
    H, W = size
    heatmap = np.zeros((H, W), dtype=np.float32)
    radius = int(sigma * 4)
    # Generate coordinates for the kernel
    y_idx, x_idx = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(x_idx**2 + y_idx**2) / (2 * sigma**2))
    
    for x, y in coords:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            # Calculate bounds for the patch in the heatmap
            y1, y2 = max(0, iy - radius), min(H, iy + radius + 1)
            x1, x2 = max(0, ix - radius), min(W, ix + radius + 1)
            
            # Calculate bounds for the kernel
            ky1, ky2 = y1 - (iy - radius), y2 - (iy - radius)
            kx1, kx2 = x1 - (ix - radius), x2 - (ix - radius)
            
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], kernel[ky1:ky2, kx1:kx2])
    
    return heatmap

class RootTipDataset(Dataset):
    def __init__(self, image_paths, annotations_dict, patch_size=512, stride=256, transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.annotations_dict = annotations_dict
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.samples = []  # (image_idx, x, y)
        self._last_img_idx = -1
        self._last_img_arr = None
        
        for img_idx, image_path in enumerate(self.image_paths):
            with Image.open(image_path) as img:
                W, H = img.size
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.samples.append((img_idx, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, x, y = self.samples[idx]
        image_path = self.image_paths[img_idx]
        
        if img_idx == self._last_img_idx:
            image = self._last_img_arr
        else:
            image = np.array(Image.open(image_path))
            if image.ndim == 2:
                image = np.stack([image]*3, axis=-1)
            self._last_img_idx = img_idx
            self._last_img_arr = image
        
        annots = self.annotations_dict.get(image_path.name, [])
        tips = [(tx - x, ty - y) for (tx, ty) in annots if x <= tx < x+self.patch_size and y <= ty < y+self.patch_size]
        heatmap = make_heatmap(tips, (self.patch_size, self.patch_size))
        patch = image[y:y+self.patch_size, x:x+self.patch_size]
        
        patch_t = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0)

        return patch_t, heatmap_t

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)

class FeatureExtractor(nn.Module):
    def __init__(self, model: UNet, layer='enc3'):
        super().__init__()
        self.model = model
        self.layer = layer
        self.features = None
        self._register_hook()

    def _register_hook(self):
        layer_module = getattr(self.model, self.layer)
        layer_module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features = output

    def forward(self, x):
        _ = self.model(x)
        return self.features
