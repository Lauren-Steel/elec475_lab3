import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import os

from load_cifar100 import get_cifar100_dataloaders
from utils import save_model, plot_loss

def get_alexnet(num_classes=100):
    model = models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def validate_model(model, val_loader, criterion, device):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    return avg_val_loss, val_accuracy



def train_alexnet(epochs=25, save_interval=5):
    alexnet_directory = 'alexnet_results'
    if not os.path.exists(alexnet_directory):
        os.makedirs(alexnet_directory)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_cifar100_dataloaders(batch_size=32, resize=224)
    model = get_alexnet()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        start_time = time.time()

        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        avg_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f} seconds.")
        print(f"Training Loss: {epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if (epoch + 1) % save_interval == 0:
            model_path = os.path.join(alexnet_directory, f'alexnet_epoch{epoch+1}.pth')
            save_model(model, model_path)
            plot_loss(train_losses, val_losses, epoch, alexnet_directory)

    final_model_path = os.path.join(alexnet_directory, 'alexnet_final.pth')
    save_model(model, final_model_path)


if __name__ == '__main__':
    train_alexnet()
