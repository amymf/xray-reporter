from torch.utils.data import Dataset
from PIL import Image
import torch

MAX_IMAGES = 2

class IUXRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, tokenizer, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        uid = row['uid']
        images = row['images']
        indication = row['indication']
        findings = row['findings']
        impression = row['impression']

        # Load images
        image_paths = [f"{self.image_dir}/{img}" for img in images[:MAX_IMAGES]]
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        # Apply transformations if any
        if self.transform:
            images = [self.transform(img) for img in images]
        # Stack images into a single tensor
        images = torch.stack(images) # (N, num_channels, H, W) where N is the number of images
        N = images.shape[0]
        if N < MAX_IMAGES:
            images = torch.cat([images, images], dim=0) # Duplicate as later we average (this assumes N=2)

        # Tokenize text data
        indication_tokens = self.tokenizer(indication, return_tensors='pt', padding='max_length', max_length=64, truncation=True)
        findings_tokens = self.tokenizer(findings, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
        impression_tokens = self.tokenizer(impression, return_tensors='pt', padding='max_length', max_length=128, truncation=True)

        return {
            'uid': uid,
            'images': images,
            'indication': indication_tokens,
            'findings': findings_tokens,
            'impression': impression_tokens
        }