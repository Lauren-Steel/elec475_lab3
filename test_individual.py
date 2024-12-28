import torch
import argparse
from train_alexnet import get_alexnet
from train_resnet18 import get_resnet18
from train_vgg16 import get_vgg16
from load_cifar100 import get_cifar100_dataloaders

def load_model(model_path, model_type, num_classes=100):
    """ Load the specified model with pretrained weights. """
    if model_type == 'alexnet':
        model = get_alexnet(num_classes)
    elif model_type == 'resnet18':
        model = get_resnet18(num_classes)
    elif model_type == 'vgg16':
        model = get_vgg16(num_classes)
    else:
        raise ValueError("Unsupported model type")

    model.load_state_dict(torch.load(model_path))
    return model

def test_model(model, device, test_loader):
    """ Evaluate the model on the test set and calculate top-1 and top-5 error rates. """
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.topk(outputs, k=5, dim=1)
            correct_top1 += (preds[:, 0] == labels).sum().item()
            correct_top5 += (preds == labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(test_loader)
    top1_error = 1 - (correct_top1 / total)
    top5_error = 1 - (correct_top5 / total)
    return avg_test_loss, top1_error, top5_error

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CIFAR-100 Test Data
    _, test_loader = get_cifar100_dataloaders(batch_size=32, resize=224)

    # Model paths and types
    model_paths = {
        'alexnet': args.alexnet_model,
        'resnet18': args.resnet18_model,
        'vgg16': args.vgg16_model
    }

    # Testing each model
    for model_type, model_path in model_paths.items():
        if model_path:

            print(f"Testing {model_type} from saved model at {model_path}")
            model = load_model(model_path, model_type)
            test_loss, top1_error, top5_error = test_model(model, device, test_loader)
            print(f"{model_type.upper()} Test Loss: {test_loss:.4f}, Top-1 Error: {top1_error:.4f}, Top-5 Error: {top5_error:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test image classification models.")
    parser.add_argument('--alexnet_model', type=str, help='Path to saved AlexNet model')
    parser.add_argument('--resnet18_model', type=str, help='Path to saved ResNet18 model')
    parser.add_argument('--vgg16_model', type=str, help='Path to saved VGG16 model')
    args = parser.parse_args()

    main(args)
