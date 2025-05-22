from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd

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
        images = row['images'][:MAX_IMAGES-1]
        indication = str(row['indication']) if not pd.isna(row['indication']) else ""
        findings = str(row['findings']) if not pd.isna(row['findings']) else ""
        impression = str(row['impression']) if not pd.isna(row['impression']) else ""


        # Load images
        image_paths = [f"{self.image_dir}/{img}" for img in images]
        images = [Image.open(img_path).convert("L") for img_path in image_paths]
        # Apply transformations if any
        if self.transform:
            images = [self.transform(img) for img in images]
        # Stack images into a single tensor
        images = torch.stack(images) # (N, num_channels, H, W) where N is the number of images
        N = images.shape[0]
        if N < MAX_IMAGES:
            padding_imgs = images[0].unsqueeze(0).repeat(MAX_IMAGES - N, 1, 1, 1) # repeat the first image
            images = torch.cat([images, padding_imgs], dim=0)


        # Tokenize text data
        # indication_tokens = self.tokenizer(indication, return_tensors='pt', padding='max_length', max_length=64, truncation=True)
        findings_tokens = self.tokenizer(findings, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
        impression_tokens = self.tokenizer(impression, return_tensors='pt', padding='max_length', max_length=128, truncation=True)

        return {
            'uid': uid,
            'images': images,
            # 'indication': indication_tokens['input_ids'].squeeze(0),
            'findings': findings_tokens['input_ids'].squeeze(0),
            'attn_mask': findings_tokens['attention_mask'].squeeze(0),
            # 'impression': impression_tokens['input_ids'].squeeze(0)
        }
    