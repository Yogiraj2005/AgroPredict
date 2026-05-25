import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image

# ==========================================
# 1. Custom Model Definition (Model A)
# ==========================================
class AgroPredictModelA(nn.Module):
    def __init__(self, num_classes):
        super(AgroPredictModelA, self).__init__()
        # Load pre-trained EfficientNetV2-S
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.efficientnet = efficientnet_v2_s(weights=weights)
        
        # Get the number of features before the final classifier layer
        in_features = self.efficientnet.classifier[1].in_features
        
        # Remove the default classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Single Head: Disease Classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.efficientnet(x)
        class_logits = self.classifier(features)
        return class_logits

# ==========================================
# 2. DSI Calculation (Image-Based Area Analysis)
# ==========================================
def calculate_dsi(image_path):
    """
    Calculates the Disease Severity Index (DSI) by analyzing the actual
    leaf image to determine what percentage of the leaf area is affected.
    
    Steps:
      1. Segment the leaf from the background using HSV green detection.
      2. Detect diseased (brown, yellow, gray, necrotic) regions on the leaf.
      3. DSI = diseased_pixels / total_leaf_pixels  (scale 0.0 to 1.0)
    
    Returns:
        dsi (float): Disease Severity Index between 0.0 (healthy) and 1.0 (fully affected)
        affected_percent (float): Percentage of leaf area affected
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --------------------------------------------------
    # Step 1: Segment the leaf from the background
    # Detect green + yellow-green regions (healthy + mildly affected leaf tissue)
    # --------------------------------------------------
    # Broad green range to capture the entire leaf
    lower_green = np.array([15, 20, 20])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Also include brown/tan regions (diseased tissue is still part of the leaf)
    lower_brown = np.array([5, 20, 30])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Include gray/necrotic regions
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 50, 180])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Combine all masks to get the full leaf region
    leaf_mask = cv2.bitwise_or(green_mask, brown_mask)
    leaf_mask = cv2.bitwise_or(leaf_mask, gray_mask)
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    
    if total_leaf_pixels == 0:
        # Could not segment leaf; return 0
        return 0.0, 0.0
    
    # --------------------------------------------------
    # Step 2: Detect HEALTHY green regions on the leaf
    # --------------------------------------------------
    # Strict green range for healthy tissue only
    lower_healthy = np.array([25, 40, 40])
    upper_healthy = np.array([90, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
    
    # Only count healthy pixels that are inside the leaf
    healthy_on_leaf = cv2.bitwise_and(healthy_mask, leaf_mask)
    healthy_pixels = cv2.countNonZero(healthy_on_leaf)
    
    # --------------------------------------------------
    # Step 3: Calculate DSI
    # Diseased area = Total leaf area - Healthy area
    # --------------------------------------------------
    diseased_pixels = total_leaf_pixels - healthy_pixels
    dsi = diseased_pixels / total_leaf_pixels
    
    # Clamp DSI between 0 and 1
    dsi = max(0.0, min(1.0, dsi))
    affected_percent = round(dsi * 100, 2)
    
    return round(dsi, 4), affected_percent

# Wrapper to apply specialized transforms to subsets
class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# ==========================================
# 2. Training Workflow
# ==========================================
def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, save_path="model_a_weights.pth", val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard transformations for EfficientNetV2
    img_size = 224
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    image_datasets = {
        'train': SubsetWrapper(train_subset, transform=train_transform),
        'val': SubsetWrapper(val_subset, transform=val_transform)
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    print(f"Classes found: {class_names}")
    print(f"Training on {train_size} images, validating on {val_size} images.")
    
    # Save class names for inference later
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    # Initialize model
    model = AgroPredictModelA(num_classes=num_classes)
    model = model.to(device)

    # Loss functions
    criterion_classification = nn.CrossEntropyLoss()
    criterion_dsi = nn.MSELoss() # Using MSE for DSI regression
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0

    # Training Loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    class_logits = model(inputs)
                    _, preds = torch.max(class_logits, 1)
                    
                    # Classification loss only (DSI is now computed via image analysis)
                    loss = criterion_classification(class_logits, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"Saving best validation model (Acc: {best_acc:.4f})")
                
        print()

    print(f'Training complete. Best val Acc: {best_acc:4f}')
    return model

# ==========================================
# 3. Inference Function (Usage in Production)
# ==========================================
def predict_image(image_path, model_path="model_a_weights.pth", class_names_path="class_names.txt"):
    """
    Loads the trained model and predicts disease + calculates DSI from actual image analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Class Names
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file {class_names_path} not found.")
    with open(class_names_path, "r") as f:
        class_names = f.read().splitlines()
        
    # Load Model
    model = AgroPredictModelA(num_classes=len(class_names))
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model weights file {model_path} not found. Please train first.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model = model.to(device)
    model.eval()
    
    # Preprocess Image for classification
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # --- Disease Classification ---
    with torch.no_grad():
        class_logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(class_logits, dim=1)
        confidence_score, predicted_idx = torch.max(probabilities, 1)
        
        disease_name = class_names[predicted_idx.item()]
        conf_value = confidence_score.item()
        disease_name = disease_name.replace('_', ' ').title()
    
    # --- DSI Calculation (Image-Based Area Analysis) ---
    if "healthy" in disease_name.lower():
        dsi_value = 0.0
        affected_percent = 0.0
    else:
        dsi_value, affected_percent = calculate_dsi(image_path)

    severity = quantify_severity(dsi_value)
            
    return {
        "disease_name": disease_name,
        "confidence_score": round(conf_value, 4),
        "dsi": dsi_value,
        "affected_area_percent": affected_percent,
        "severity": severity
    }

# ==========================================
# Model B - Severity Quantification
# ==========================================
def quantify_severity(dsi):
    """
        Model B: Maps the numeric Disease Severity Index (0.0 to 1.0) into qualitative categories.
        Thresholds:
        - Healthy: DSI == 0.0
        - Mild:    0.0 < DSI < 0.3
        - Moderate: 0.3 <= DSI < 0.7
        - Severe:  DSI >= 0.7
    """
    if dsi <= 0.0:
        return "Healthy"
    elif dsi < 0.3:
        return "Mild"
    elif dsi < 0.7:
        return "Moderate"
    else:
        return "Severe"

# ==========================================
# Usage instructions
# ==========================================
if __name__ == "__main__":
    print("Welcome to Model A (EfficientNetV2) & Model B")

    # --- Training Workflow ---
    # dataset_directory = "maize_R" 
    # if os.path.exists(dataset_directory):
    #     train_model(data_dir=dataset_directory, num_epochs=15, batch_size=16)
    # else:
    #     print(f"Dataset directory '{dataset_directory}' not found. Please verify the path.")
    
    # --- Inference Workflow ---

    # test_img = "test_1-DM.jpg"
    # test_img = "test_2-MLN.jpg"
    # test_img = "test_3-CR.jpg"
    test_img = "test_4-B.jpg"
    
    if os.path.exists(test_img) and os.path.exists("model_a_weights.pth"):
        result = predict_image(test_img)
        # print("Prediction Result:")
        # print(result)
        
        print("\nPath A Results:")
        print(f"  Disease Name     : {result['disease_name']}")
        print(f"  Confidence Score : {result['confidence_score']:.4f}")
        print(f"  DSI (0-1)        : {result['dsi']:.4f}")
        print(f"  Affected Area    : {result['affected_area_percent']}%")
        print(f"  Severity         : {result['severity']}")