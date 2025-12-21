import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from src.utils.common import get_timestamp, filter_outlier_images

class AffinityMLP(nn.Module):
    def __init__(self, feature_dim=256):
        super(AffinityMLP, self).__init__()
        # Input: [FeatA (256), FeatB (256), dx, dy, dist] = 515
        input_dim = feature_dim * 2 + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class LinkingDataset(Dataset):
    def __init__(self, tip_features, tip_links, mode='train', outlier_threshold=2.0):
        """
        tip_features: dict mapping image_basename -> list of tip dicts (x, y, features)
        tip_links: dict mapping "img1->img2" basenames -> list of {tip1_index, tip2_index}
        """
        self.samples = []
        self.mode = mode
        
        all_imgs = sorted(tip_features.keys(), key=get_timestamp)
        if not all_imgs:
            return

        valid_imgs = filter_outlier_images(all_imgs, tip_features, outlier_threshold)

        for link_key, links in tip_links.items():
            img1_name, img2_name = link_key.split("->")
            if img1_name not in valid_imgs or img2_name not in valid_imgs:
                continue
            if img1_name not in tip_features or img2_name not in tip_features:
                continue
                
            tips1 = tip_features[img1_name]
            tips2 = tip_features[img2_name]
            
            true_links = set((l['tip1_index'], l['tip2_index']) for l in links)
            
            # Positives
            for t1_idx, t2_idx in true_links:
                if t1_idx < len(tips1) and t2_idx < len(tips2):
                    self.samples.append((tips1[t1_idx], tips2[t2_idx], 1.0))
            
            # Negatives
            for t1_idx in range(len(tips1)):
                true_t2_idx = next((l[1] for l in true_links if l[0] == t1_idx), -1)
                potential_negs = [j for j in range(len(tips2)) if j != true_t2_idx]
                if potential_negs:
                    num_negs = min(2, len(potential_negs))
                    negs = np.random.choice(potential_negs, num_negs, replace=False)
                    for t2_idx in negs:
                        self.samples.append((tips1[t1_idx], tips2[t2_idx], 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tip1, tip2, label = self.samples[idx]
        
        f1 = np.array(tip1['features'])
        f2 = np.array(tip2['features'])
        
        dx = tip2['x'] - tip1['x']
        dy = tip2['y'] - tip1['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        spatial = np.array([dx/100.0, dy/100.0, dist/100.0])
        x = np.concatenate([f1, f2, spatial], axis=0)
        return torch.from_numpy(x).float(), torch.tensor([label]).float()
