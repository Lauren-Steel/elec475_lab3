import torch
import os
import argparse
from ensemble_methods import ensemble_max_probability, ensemble_probability_averaging, ensemble_majority_voting
from train_alexnet import get_alexnet
from train_resnet18 import get_resnet18
from train_vgg16 import get_vgg16
from load_cifar100 import get_cifar100_dataloaders  


def load_model(model_path, model_type, num_classes=100, device=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_type == 'alexnet':
        model = get_alexnet(num_classes)
    elif model_type == 'resnet18':
        model = get_resnet18(num_classes)
    elif model_type == 'vgg16':
        model = get_vgg16(num_classes)
    else:
        raise ValueError("Unsupported model type")
    
    model.load_state_dict(torch.load(model_path, map_location=device))  # Ensure model is loaded to the right device
    model.to(device)  # Move model to the specified device
    model.eval()
    return model


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models = []
    if args.alexnet_model:
        alexnet = load_model(args.alexnet_model, 'alexnet', num_classes=100, device=device)
        models.append(alexnet)
    if args.resnet18_model:
        resnet18 = load_model(args.resnet18_model, 'resnet18', num_classes=100, device=device)
        models.append(resnet18)
    if args.vgg16_model:
        vgg16 = load_model(args.vgg16_model, 'vgg16', num_classes=100, device=device)
        models.append(vgg16)

    if not models:
        raise ValueError("No models were provided for ensemble testing.")
    
    # Load the CIFAR-100 test data
    _, test_loader = get_cifar100_dataloaders(batch_size=32, resize=224)

    # Ensemble Testing
    top1_accuracy, top5_accuracy = ensemble_max_probability(models, test_loader, device)
    print(f"Max Probability: Top-1 Accuracy = {top1_accuracy*100:.2f}%, Top-5 Accuracy = {top5_accuracy*100:.2f}%")

    top1_accuracy, top5_accuracy = ensemble_probability_averaging(models, test_loader, device)
    print(f"Probability Averaging: Top-1 Accuracy = {top1_accuracy*100:.2f}%, Top-5 Accuracy = {top5_accuracy*100:.2f}%")

    top1_accuracy, top5_accuracy = ensemble_majority_voting(models, test_loader, device)
    print(f"Majority Voting: Top-1 Accuracy = {top1_accuracy*100:.2f}%, Top-5 Accuracy = {top5_accuracy*100:.2f}%")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test image classification models using ensemble methods")
    parser.add_argument('--alexnet_model', type=str, help='Path to saved AlexNet model')
    parser.add_argument('--resnet18_model', type=str, help='Path to saved ResNet18 model')
    parser.add_argument('--vgg16_model', type=str, help='Path to saved VGG16 model')
    args = parser.parse_args()

    main(args)
