import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
import os


# Get the names of the flowers from the cloud
print("Connecting to dataset...")
dataset = load_dataset("nsarker/flower-detection", split='train')
dataset2 = load_dataset("nkirschi/oxford-flowers")
label_names = dataset.features['label'].names
print(f" Flowers we can detect: {label_names}")
print(dataset2)



# Load the Brain (MobileNetV2)
print(" Loading the Brain...")
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Save a test image from the dataset
print(" Saving test_daisy.png...")
dataset[0]['image'].save("test_daisy.png")

# RUN THE PREDICTION
print(" Analyzing image...")
img = Image.open("test_daisy.png").convert("RGB")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

with torch.no_grad():
    output = model(batch_t)

# Find the item
prob = torch.nn.functional.softmax(output[0], dim=0)
conf, index = torch.max(prob, 0)

print("-" * 30)
print(f"RESULT: The model is {conf.item()*100:.1f}% sure this is ID #{index.item()}")
print("-" * 30)