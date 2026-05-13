A#!/usr/bin/env python3
"""
train.py: Production-grade training script for Flower Recognition using MobileNetV2.
Includes advanced metrics (Loss/Accuracy plots) and Confusion Matrix generation.
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OxfordFlowersDataset(Dataset):
    """PyTorch Dataset for Oxford Flowers 102."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = int(item["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms() -> Dict[str, transforms.Compose]:
    """Returns training and validation transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    }

def setup_model(num_classes: int = 102, device: torch.device = torch.device("cpu")) -> nn.Module:
    """Initializes MobileNetV2 for transfer learning."""
    logger.info("Initializing MobileNetV2...")
    
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.mobilenet_v2(pretrained=True)
    
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Modify classifier head
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # Fine-tuning: Unfreeze the last few blocks
    for block in model.features[15:]:
        for param in block.parameters():
            param.requires_grad = True
            
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

def plot_metrics(history: Dict[str, List[float]], output_dir: str):
    """Generates and saves loss and accuracy plots."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_report.png")
    plt.savefig(plot_path)
    logger.info(f"Training metrics plot saved to {plot_path}")
    plt.close()

def generate_confusion_matrix(model, loader, device, label_mapping, output_dir):
    """Generates a confusion matrix for the final model."""
    logger.info("Generating Confusion Matrix...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting
    plt.figure(figsize=(20, 15))
    # We only show top N most frequent classes if 102 is too many to read
    # For now, let's plot the full matrix but with smaller font
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    plt.close()

def train(args):
    # Device selection (optimized for Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Create directories
    os.makedirs(args.model_output, exist_ok=True)
    os.makedirs(args.mapping_output, exist_ok=True)
    os.makedirs(args.reports_output, exist_ok=True)

    # Load Dataset
    ds = load_dataset("nkirschi/oxford-flowers")
    
    # Labels Logic (Fixed duplication)
    id_to_name = {
        "1": "pink primrose", "2": "hard-leaved pocket orchid", "3": "canterbury bells", "4": "sweet pea", "5": "english marigold",
        "6": "tiger lily", "7": "moon orchid", "8": "bird of paradise", "9": "monkshood", "10": "globe thistle",
        "11": "snapdragon", "12": "colts foot", "13": "king protea", "14": "spear thistle", "15": "yellow iris",
        "16": "globe-flower", "17": "purple coneflower", "18": "peruvian lily", "19": "balloon flower", "20": "giant white arum lily",
        "21": "fire lily", "22": "pincushion flower", "23": "fritillary", "24": "red ginger", "25": "grape hyacinth",
        "26": "corn poppy", "27": "prince of wales feathers", "28": "stemless gentian", "29": "artichoke", "30": "sweet william",
        "31": "carnation", "32": "garden phlox", "33": "love in the mist", "34": "mexican aster", "35": "alpine sea holly",
        "36": "ruby-lipped cattleya", "37": "cape flower", "38": "great masterwort", "39": "siam tulip", "40": "lenten rose",
        "41": "barbeton daisy", "42": "daffodil", "43": "sword lily", "44": "poinsettia", "45": "bolero deep blue",
        "46": "wallflower", "47": "marigold", "48": "buttercup", "49": "oxeye daisy", "50": "common dandelion",
        "51": "mexican sunflower", "52": "rose", "53": "california poppy", "54": "osteospermum", "55": "spring crocus",
        "56": "bearded iris", "57": "windflower", "58": "tree poppy", "59": "gazania", "60": "azalea",
        "61": "water lily", "62": "rose mallow", "63": "cardoon", "64": "freesia", "65": "common tulip",
        "66": "wild pansy", "67": "primula", "68": "sunflower", "69": "pelargonium", "70": "bishop of llandaff",
        "71": "gaura", "72": "geranium", "73": "orange dahlia", "74": "pink-yellow dahlia", "75": "thorn apple",
        "76": "morning glory", "77": "passion flower", "78": "lotus", "79": "toad lily", "80": "anthurium",
        "81": "frangipani", "82": "clematis", "83": "hibiscus", "84": "columbine", "85": "desert-rose",
        "86": "tree mallow", "87": "magnolia", "88": "cyclamen", "89": "cactus dahlia", "90": "gloxinia",
        "91": "bee balm", "92": "ball moss", "93": "foxglove", "94": "bougainvillea", "95": "camellia",
        "96": "mallow", "97": "mexican petunia", "98": "bromelia", "99": "blanket flower", "100": "trumpet creeper",
        "101": "blackberry lily", "102": "canna lily"
    }

    label_ids = ds["train"].features["label"].names
    label_mapping = {i: id_to_name.get(label_id, f"Flower {label_id}") for i, label_id in enumerate(label_ids)}
    
    with open(os.path.join(args.mapping_output, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=4)

    # Dataloaders
    ts = get_transforms()
    val_key = "validation" if "validation" in ds else "test"
    train_loader = DataLoader(OxfordFlowersDataset(ds["train"], ts["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(OxfordFlowersDataset(ds[val_key], ts["val"]), batch_size=args.batch_size)

    # Model and Training Configuration
    model = setup_model(num_classes=len(label_mapping), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # History Tracking
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_names': label_mapping
            }, os.path.join(args.model_output, "flower_model_v2.pth"))
            logger.info(f"Saved best model (Acc: {val_acc:.4f})")

    # Post-Training Reports
    plot_metrics(history, args.reports_output)
    generate_confusion_matrix(model, val_loader, device, label_mapping, args.reports_output)
    logger.info("Training complete. Check the 'reports/' folder for charts.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-output", type=str, default=os.path.join(BASE_DIR, "models"))
    parser.add_argument("--mapping-output", type=str, default=os.path.join(BASE_DIR, "data", "mappings"))
    parser.add_argument("--reports-output", type=str, default=os.path.join(BASE_DIR, "reports"))
    
    train(parser.parse_args())
