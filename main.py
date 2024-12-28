import os
import argparse
from train_alexnet import train_alexnet
from train_resnet18 import train_resnet18
from train_vgg16 import train_vgg16
import traceback

def main(args):
    print(f"args {args}")
    if args.alexnet:
        print("Starting training AlexNet...")
        try:
            train_alexnet()  # Train AlexNet until completion
            print("AlexNet training completed.")
        except Exception as e:
            print("An error occurred while training AlexNet:")
            traceback.print_exc() 

    if args.resnet18:
        print("Starting training ResNet18...")
        try:
            train_resnet18()  # Train ResNet18 until completion
            print("ResNet18 training completed.")
        except Exception as e:
            print("An error occurred while training ResNet18:")
            traceback.print_exc()  

    if args.vgg16:
        print("Starting training VGG16...")
        try:
            train_vgg16()  # Train VGG16 until completion
            print("VGG16 training completed.")
        except Exception as e:
            print("An error occurred while training VGG16:")
            traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train image classification models.")
    parser.add_argument('--alexnet', action='store_true', help='Train the AlexNet model')
    parser.add_argument('--resnet18', action='store_true', help='Train the ResNet18 model')
    parser.add_argument('--vgg16', action='store_true', help='Train the VGG16 model')
    args = parser.parse_args()

    # If no models are specified, train both by default
    if not (args.alexnet or args.resnet18 or args.vgg16):
        args.alexnet = args.resnet18 = args.vgg16 = True

    main(args)
