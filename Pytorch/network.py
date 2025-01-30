import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os

from model import Model


def create_model(epochs: int, learning_rate: float =0.001, save: bool =False) -> Model:
    # Remove previous model
    if os.path.exists("model.pth"):
        os.remove("model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [-1, 1]
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


    model = Model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ### TRAINING

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            # Loss function
            loss = criterion(outputs, labels)
            # Reset gradients
            optimizer.zero_grad()
            # Calculates gradients
            loss.backward()
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.3f}, Accuracy {correct / total:.3f}")

    ### TESTING

    model.eval()
    correct = 0
    total = 0
    # Skips calculating of the gradients
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {correct / total:.3f}")

    if save:
        torch.save(model.state_dict(), "model.pth")

    return model


def load_model(device) -> Model:
    # Model doesn't exist
    if not os.path.exists("model.pth"):
        return None
    
    print(f"Using {device} device")

    model = Model()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model
