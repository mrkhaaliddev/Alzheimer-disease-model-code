import argparse
import os
import torch
from alzheimer_classifier_training import AlzheimerClassifier, AlzheimerDataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths to data
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'

# Helper to evaluate accuracy
# New helper to load all images from both train and test as validation set

def load_all_images_for_validation(classifier, train_dir, test_dir):
    train_paths, train_labels = classifier._load_folder_data(train_dir)
    test_paths, test_labels = classifier._load_folder_data(test_dir)
    # Concatenate all
    all_paths = np.concatenate([train_paths, test_paths])
    all_labels = np.concatenate([train_labels, test_labels])
    return all_paths, all_labels


def evaluate_model_on_val(model_path, device):
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if not (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'model_architecture' in checkpoint):
        print(f"❌ Model file {model_path} does not contain architecture info. Please use a complete model file.")
        return None, None, None
    model_arch = checkpoint['model_architecture']
    # Map possible names to build_model argument
    arch_map = {
        'ResNet': 'resnet50',
        'EfficientNet': 'efficientnet_b3',
        'Inception3': 'inception_v3',
        'resnet50': 'resnet50',
        'efficientnet_b3': 'efficientnet_b3',
        'inception_v3': 'inception_v3',
    }
    model_name = arch_map.get(model_arch, None)
    if model_name is None:
        print(f"❌ Unknown model architecture '{model_arch}' in {model_path}")
        return None, None, None
    # Initialize classifier and build correct model
    classifier = AlzheimerClassifier(device=device)
    classifier.build_model(model_name)
    if classifier.model is not None:
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.model.eval()
    else:
        print(f"❌ Failed to build model for architecture '{model_arch}'")
        return None, None, None

    # Load all images from both train and test as validation set
    val_paths, val_labels = load_all_images_for_validation(classifier, TRAIN_DIR, TEST_DIR)
    val_dataset = AlzheimerDataset(val_paths, val_labels, classifier.get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=classifier.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Evaluate on validation set with progress bar
    correct = 0
    total = 0
    print(f"Evaluating model on ALL images from train and test ({len(val_dataset)} samples)...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Evaluating {os.path.basename(model_path)}", unit="batch")
        for i, (data, target) in enumerate(pbar, 1):
            data, target = data.to(device), target.to(device)
            output = classifier.model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            percent = 100. * (i / len(val_loader))
            pbar.set_postfix({"Progress": f"{percent:.1f}%", "Correct": correct, "Total": total})
    val_acc = 100. * correct / total if total > 0 else 0.0
    return val_acc, total, model_arch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model(s) accuracy on validation set (auto-detects architecture)')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='Path(s) to model .pth files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
        print(f"\nEvaluating model: {model_path}")
        val_acc, total, model_arch = evaluate_model_on_val(model_path, device)
        if val_acc is not None:
            print(f"Model Architecture: {model_arch}")
            print(f"Validation Accuracy: {val_acc:.2f}% (on {total} samples)") 