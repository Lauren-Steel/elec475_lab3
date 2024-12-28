import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from load_cifar100 import get_cifar100_dataloaders
from utils import save_model, plot_loss


def load_model(model_path, model_type, num_classes=100):
    if model_type == 'alexnet':
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path))
    return model



def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.topk(5, 1, True, True)
            correct_top1 += torch.eq(preds[:, 0], labels).sum().item()
            correct_top5 += torch.eq(preds, labels.view(-1, 1).expand_as(preds)).sum().item()
            total += labels.size(0)
    
    top1_error = 1 - correct_top1 / total
    top5_error = 1 - correct_top5 / total
    return top1_error, top5_error

if __name__ == '__main__':
    _, test_loader = get_cifar100_dataloaders(batch_size=64, resize=224)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Test AlexNet
    model_path = 'alexnet_epoch5_parallel.pth'  # Modify as needed
    alexnet = load_model(model_path, 'alexnet')
    top1_error, top5_error = test_model(alexnet, test_loader, device)
    print(f"AlexNet Top-1 Error: {top1_error:.4f}, Top-5 Error: {top5_error:.4f}")

    # Test Resnet
    model_path = 'resnet18_epoch5_parallel.pth'  # Modify as needed
    alexnet = load_model(model_path, 'resnet18')
    top1_error, top5_error = test_model(alexnet, test_loader, device)
    print(f"ResNet18 Top-1 Error: {top1_error:.4f}, Top-5 Error: {top5_error:.4f}")

    # Test Resnet
    model_path = 'vgg16_epoch5_parallel.pth'  # Modify as needed
    alexnet = load_model(model_path, 'vgg16')
    top1_error, top5_error = test_model(alexnet, test_loader, device)
    print(f"VGG16 Top-1 Error: {top1_error:.4f}, Top-5 Error: {top5_error:.4f}")

