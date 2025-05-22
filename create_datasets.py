import pandas as pd
from dataset import IUXRayDataset
from transformers import GPT2Tokenizer
from torchvision import transforms
import torch
import ast

df = pd.read_csv('iuxray_dataset.csv')
image_dir = '../.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2_prepared")
transform = transforms.Compose([
    transforms.Resize(256), # resize short edge to 256 pixels
    transforms.CenterCrop(224), # crop to 224x224
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # normalize to [-1, 1] range for grayscale
])

df['images'] = df['images'].apply(ast.literal_eval)  # Convert string representation of list to actual list
df['indication'] = df['indication'].fillna('').astype(str)
df['findings'] = df['findings'].fillna('').astype(str)
df['impression'] = df['impression'].fillna('').astype(str)

dataset = IUXRayDataset(df, image_dir, tokenizer, transform=transform)

train_size = int(0.8 * len(dataset))
train_dataset, rest = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
val_size = int(0.5 * len(rest))
test_dataset, val_dataset = torch.utils.data.random_split(rest, [val_size, len(rest) - val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")