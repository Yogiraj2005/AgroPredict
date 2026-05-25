import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
from pathlib import Path

# Add the parent directory (Model_A) to sys.path to find model_a.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model_a import AgroPredictModelA, SubsetWrapper
from tqdm import tqdm

def find_file(filename):
    """Checks current and parent directory for a file."""
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

def evaluate_model(data_dir="maize_R", model_path="model_a_weights.pth", class_names_path="class_names.txt", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Class Names
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file {class_names_path} not found.")
    with open(class_names_path, "r") as f:
        class_names = f.read().splitlines()
    num_classes = len(class_names)

    # 2. Load Model
    model = AgroPredictModelA(num_classes=num_classes)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file {model_path} not found.")
    
    # Load weights with strict=False to handle potential architecture changes (like the DSI head removal)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model = model.to(device)
    model.eval()

    # 3. Prepare Dataset (Same split logic as training)
    img_size = 224
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir)
    val_split = 0.2
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Use fixed seed for reproducibility of the validation set
    _, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_dataset = SubsetWrapper(val_subset, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 4. Run Inference
    all_preds = []
    all_labels = []

    print(f"Running evaluation on {val_size} images...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*30)
    print("      MODEL A EVALUATION")
    print("="*30)
    print(f"Overall Accuracy: {accuracy * 100:.2f} %")
    print("\nClassification Report:")
    print(report)

    # 6. Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Model A')
    
    cm_path = "evaluate/model_a_confusion_matrix-L.png"
    plt.savefig(cm_path)
    print(f"\nConfusion Matrix saved to: {cm_path}")
    
    # Also print raw CM for quick check
    print("\nConfusion Matrix (Raw):")
    print(cm)

    # with open("evaluate/model_a_report.txt", "w") as f:
    #     f.write(f"Overall Accuracy: {accuracy:.4f}\n" + "\nClassification Report:\n" 
    #     + report + "\n" + "\nConfusion Matrix (Raw):\n" + str(cm))

if __name__ == "__main__":
    # Check if maize_R or maize exists
    target_dir = find_file("maize_R") if os.path.exists(find_file("maize_R")) else find_file("maize")
    evaluate_model(
        data_dir=target_dir,
        model_path=find_file("model_a_weights.pth"),
        class_names_path=find_file("class_names.txt")
    )
