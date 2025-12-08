from torch.utils.data import Dataset
from PIL import Image
import os

class TrickSkiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform

        # Sort folders so train & test label order matches
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Create stable label mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            label = self.class_to_idx[cls]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
