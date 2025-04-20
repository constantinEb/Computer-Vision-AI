import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from modified_resnet50 import ModifiedResNet50

#################################
## 1. SAVING AND LOADING MODEL ##
#################################

def save_model(model, path='modified_resnet50.pth'):
    """Save the entire model architecture and weights"""
    torch.save(model, path)
    print(f"Full model saved to {path}")


def save_model_weights(model, path='modified_resnet50_weights.pth'):
    """Save only the model weights (recommended approach)"""
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model(path='modified_resnet50.pth'):
    """Load the entire model"""
    model = torch.load(path)
    model.eval()  # Set to evaluation mode
    return model


def load_model_weights(num_classes=1000, path='modified_resnet50_weights.pth'):
    """Load just the model weights into a new model instance"""
    model = ModifiedResNet50(num_classes=num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    return model


#######################
## 2. TRAINING MODEL ##
#######################

def train_model(model, train_dir, val_dir=None, num_classes=1000, batch_size=32,
                num_epochs=10, learning_rate=0.001, save_path='modified_resnet50_weights.pth'):
    """
    Train the modified ResNet50 model

    Args:
        model: ModifiedResNet50 model instance
        train_dir: Directory containing training images organized in class folders
        val_dir: Directory containing validation images organized in class folders
        num_classes: Number of classes for classification
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        save_path: Path to save the model weights
    """
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if val_dir:
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        val_loader = None

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require gradients (non-frozen)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate average validation loss and accuracy
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Adjust learning rate
            scheduler.step(val_loss)

            # Save model if validation loss improved
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}, saving model...")
                best_val_loss = val_loss
                save_model_weights(model, save_path)

    # Save final model if no validation was performed
    if not val_loader:
        save_model_weights(model, save_path)

    return model


#############################
## 3. USING TRAINED MODEL ##
#############################

def preprocess_image(image_path):
    """Preprocess an image for inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def inference(model, image_path, class_names):
    """
    Run inference on a single image

    Args:
        model: Trained ModifiedResNet50 model
        image_path: Path to the image file
        class_names: List of class names

    Returns:
        predicted_class_idx: Index of the predicted class
        confidence: Confidence score (probability)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get predicted class and confidence
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    # Convert to Python types
    predicted_class_idx = predicted_class_idx.item()
    confidence = confidence.item()


    # Print results
    print(f"Image: {image_path}")

    if class_names and predicted_class_idx < len(class_names):
        print(f"Predicted class: {class_names[predicted_class_idx]}")
    else:
        print(f"Predicted class index: {predicted_class_idx}")

    print(f"Confidence: {confidence:.4f}\n")

    return predicted_class_idx, confidence


def batch_inference(model, image_dir, class_names, batch_size=16):
    """
    Run inference on a directory of images

    Args:
        model: Trained ModifiedResNet50 model
        image_dir: Directory containing images (can be organized in subdirectories)
        class_names: List of class names
        batch_size: Batch size for inference

    Returns:
        results: Dictionary mapping image paths to (class_idx, confidence) tuples
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and loader
    dataset = datasets.ImageFolder(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get file paths from dataset
    image_paths = [dataset.samples[i][0] for i in range(len(dataset))]

    y_test = [path.split('\\')[-2] for path in image_paths]
    y_pred = []
    idx = 0

    with torch.no_grad():
        for inputs, _ in tqdm(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get predicted classes and confidences
            confidences, predicted_classes = torch.max(probabilities, 1)
            # print(class_names[0])
            # print(len(predicted_classes), len(y_test), inputs.shape, outputs.shape, predicted_classes[0], class_names[int(predicted_classes[0].item())])
            # Store results
            for i in range(inputs.size(0)):
                if idx < len(image_paths):
                    y_pred.append(class_names[predicted_classes[0].item()])
                    idx += 1

    return y_test, y_pred

def main():
    # Create a new model instance
    #model = ModifiedResNet50(num_classes=3)

    # Train the model
    #train_model(model,
               # train_dir='../data_split/train_balanced',
               # val_dir='../data_split/test',
               # num_classes=3,
               # batch_size=32,
               # num_epochs=10)

    # Load a trained model
    model = load_model_weights(num_classes=3, path='modified_resnet50_weights.pth')

    class_names = {
        0: "Burrito",
        1: "Hot dog",
        2: "Muffin"
    }

    # Run inference
    inference(model, f'../additional_images/hot_dog.jpeg', class_names)
    inference(model, f'../additional_images/hot_dog_1.jpeg', class_names)
    inference(model, f'../additional_images/muffin.jpeg', class_names)
    inference(model, f'../additional_images/muffin_1.jpeg', class_names)
    inference(model, f'../additional_images/burrito.jpeg', class_names)
    inference(model, f'../additional_images/burrito_1.jpeg', class_names)

    # Batch inference
    y_test, y_pred = batch_inference(model, '../data_split/test', class_names)

    correct = sum(1 for true, pred in zip(y_test, y_pred) if true == pred)
    total = len(y_test)
    accuracy = correct / total

    # Create confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=['Burrito', 'Hot dog', 'Muffin']
    )

    # Create a figure for the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Burrito', 'Hot dog', 'Muffin'],
        yticklabels=['Burrito', 'Hot dog', 'Muffin']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2%}')

    # Save the confusion matrix as an image
    output_filename = f"confusion_matrix_modified_resnet50.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    main()