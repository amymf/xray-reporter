import pandas as pd
from dataset import IUXRayDataset
from transformers import GPT2Tokenizer
from PIL import Image
from torchvision import transforms
import torch

df = pd.read_csv('iuxray_dataset.csv')
image_dir = '../.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2_prepared")
transform = transforms.Compose([
    transforms.Resize(256), # resize short edge to 256 pixels
    transforms.CenterCrop(224), # crop to 224x224
    transforms.Grayscale(num_output_channels=3), # convert to 3 channels, duplicate grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # ImageNet dist
])

dataset = IUXRayDataset(df, image_dir, tokenizer, transform=transform)

train_size = int(0.8 * len(dataset))
train_dataset, rest = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
val_size = int(0.5 * len(rest))
test_dataset, val_dataset = torch.utils.data.random_split(rest, [val_size, len(rest) - val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")