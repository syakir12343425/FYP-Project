"""
DIV2K dataset loader.

Uses RANDOM CROP instead of resizing whole images to 1024x1024 (the proposal's
Section 3.4.3 step 1). DIV2K images are not square, so forcing them to
1024x1024 distorts aspect ratio — cropping avoids that and also gives you
more effective training variety per image (different crop each epoch).

Download DIV2K first:
  wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
  unzip into data/DIV2K_train_HR/ and data/DIV2K_valid_HR/
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class DIV2KStegoDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 512, message_bits: int = 1024, train: bool = True):
        self.root_dir = Path(root_dir)
        self.paths = sorted(
            list(self.root_dir.glob("*.png")) + list(self.root_dir.glob("*.jpg"))
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {root_dir}. Download DIV2K first — see "
                f"this file's module docstring for the wget commands."
            )
        self.crop_size = crop_size
        self.message_bits = message_bits
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        w, h = img.size
        cs = self.crop_size

        if w < cs or h < cs:
            # upscale small images rather than crash — DIV2K is normally
            # >=2K on the short side, but guard anyway
            scale = cs / min(w, h) + 0.01
            img = img.resize((int(w * scale) + 1, int(h * scale) + 1), Image.BICUBIC)
            w, h = img.size

        if self.train:
            x0 = random.randint(0, w - cs)
            y0 = random.randint(0, h - cs)
        else:
            x0 = (w - cs) // 2
            y0 = (h - cs) // 2
        img = img.crop((x0, y0, x0 + cs, y0 + cs))

        if self.train and random.random() < 0.5:
            img = TF.hflip(img)

        cover = TF.to_tensor(img)  # [0,1], (3, cs, cs)
        message = torch.randint(0, 2, (self.message_bits,)).float()
        return cover, message
