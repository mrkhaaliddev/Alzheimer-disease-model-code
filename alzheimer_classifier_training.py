import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

class AlzheimerDataset(Dataset):
    """
    Custom Dataset class for Alzheimer's disease brain images
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (augmentation + preprocessing)
        if self.transform:
            image = self.transform(image=image)['image']
        
        label = self.labels[idx]
        
        return image, label

class AlzheimerClassifier:
    def __init__(self, img_size=224, batch_size=64, num_classes=4, device=None):
        """
        Professional Alzheimer's Disease Classifier using PyTorch
        
        Args:
            img_size: Input image size
            batch_size: Training batch size
            num_classes: Number of classes (4 Alzheimer's stages)
            device: Device to use (cuda/cpu)
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.class_names = ['No Impairment', 'Very Mild Impairment', 'Mild Impairment', 'Moderate Impairment']
        self.class_mapping = {
            'No Impairment': 0,
            'Very Mild Impairment': 1,
            'Mild Impairment': 2,
            'Moderate Impairment': 3
        }
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        print(f"Using device: {self.device}")
    
    def get_preprocessing_transforms(self):
        """
        Get preprocessing transforms (applied to all images)
        """
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def get_train_transforms(self):
        """
        Get training transforms with lighter augmentation for better training accuracy
        """
        return A.Compose([
            # Light augmentations only (removed heavy distortions)
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=10, p=0.3),  # Reduced from 20 to 10
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),  # Reduced limits
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # Reduced from 0.2
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),  # Reduced noise
            # Removed: A.Blur, A.ElasticTransform, A.GridDistortion, A.OpticalDistortion
            # Preprocessing
            A.Resize(self.img_size, self.img_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def get_val_transforms(self):
        """
        Get validation transforms (no augmentation, only preprocessing)
        """
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def load_data_from_folders(self, train_dir, test_dir):
        """
        Load data directly from train/test folder structure
        """
        print("Loading data from train/test folders...")
        
        # Load training data
        train_paths, train_labels = self._load_folder_data(train_dir)
        print(f"Training samples: {len(train_paths)}")
        
        # Load test data
        test_paths, test_labels = self._load_folder_data(test_dir)
        print(f"Test samples: {len(test_paths)}")
        
        # Split training data into train/validation (80/20)
        from sklearn.model_selection import train_test_split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        print(f"Final split:")
        print(f"  Training: {len(train_paths)} samples")
        print(f"  Validation: {len(val_paths)} samples")
        print(f"  Test: {len(test_paths)} samples")
        
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    
    def _load_folder_data(self, data_dir):
        """
        Load data from a single directory with class subdirectories
        """
        image_paths = []
        labels = []
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist!")
                continue
            
            print(f"Loading {class_name} from {class_dir}...")
            
            # Get all images in this class directory
            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(class_dir, filename)
                    image_paths.append(image_path)
                    labels.append(self.class_mapping[class_name])
            
            print(f"  Found {len([l for l in labels if l == self.class_mapping[class_name]])} images")
        
        return image_paths, np.array(labels)
    
    def create_data_loaders(self, train_paths, train_labels, val_paths, val_labels, test_paths, test_labels):
        """
        Create data loaders for training, validation, and testing
        """
        # Get transforms
        train_transform = self.get_train_transforms()
        val_transform = self.get_val_transforms()
        
        # Create datasets
        train_dataset = AlzheimerDataset(train_paths, train_labels, train_transform)
        val_dataset = AlzheimerDataset(val_paths, val_labels, val_transform)
        test_dataset = AlzheimerDataset(test_paths, test_labels, val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, model_name='resnet50'):
        """
        Build the classification model using transfer learning
        """
        print(f"Building {model_name.upper()} model architecture...")
        
        if model_name == 'resnet50':
            # Use ResNet-50 (most stable and widely used)
            model = models.resnet50(weights='IMAGENET1K_V2')
            num_features = model.fc.in_features
            
            # Replace the final layer
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, self.num_classes)
            )
            
        elif model_name == 'efficientnet_b3':
            # Use EfficientNet-B3
            model = models.efficientnet_b3(weights='IMAGENET1K_V1')
            num_features = model.classifier[1].in_features
            
            # Replace classifier
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, self.num_classes)
            )
            
        elif model_name == 'inception_v3':
            # Use InceptionV3
            model = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=False)
            num_features = model.fc.in_features
            
            model.fc = nn.Sequential(
                nn.Dropout(0.2),  # Reduced from 0.3
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),  # Reduced from 0.5
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),  # Reduced from 0.3
                nn.Linear(256, self.num_classes)
            )
        
        # Move model to device
        model = model.to(self.device)
        
        self.model = model
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion, epoch):
        """
        Validate for one epoch
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, train_loader, val_loader, epochs=30, learning_rate=0.002):
        """
        Train the model with two-stage training and improved learning rate schedule
        """
        print("Starting model training...")
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Stage 1: Train with frozen backbone
        print("\n=== Stage 1: Training with frozen backbone ===")
        
        # Freeze backbone
        if hasattr(self.model, 'features'):
            for param in self.model.features.parameters():
                param.requires_grad = False
        elif hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            # For models like ResNet, freeze all except final layer
            for name, param in self.model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
        
        # Optimizer for stage 1 with higher learning rate
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=learning_rate, weight_decay=1e-4)
        # Better scheduler: warmup + cosine decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/100)
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        # Stage 1 training
        stage1_epochs = min(20, epochs // 2)
        for epoch in range(stage1_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion, epoch)
            
            # Count predictions per class on validation set
            self.model.eval()
            val_pred_counts = [0] * self.num_classes
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    output = self.model(data)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    for p in preds:
                        val_pred_counts[p] += 1
            print(f"Validation predictions per class: {dict(zip(self.class_names, val_pred_counts))}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_alzheimer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}/{stage1_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)
        
        # Stage 2: Fine-tune with unfrozen backbone
        print("\n=== Stage 2: Fine-tuning with unfrozen backbone ===")
        
        # Calculate remaining epochs first
        remaining_epochs = epochs - stage1_epochs
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # New optimizer with lower learning rate for fine-tuning
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate/20, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=learning_rate/1000)
        
        # Reset patience
        patience_counter = 0
        
        # Stage 2 training
        for epoch in range(remaining_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, stage1_epochs + epoch)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion, stage1_epochs + epoch)
            
            # Count predictions per class on validation set
            self.model.eval()
            val_pred_counts = [0] * self.num_classes
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    output = self.model(data)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    for p in preds:
                        val_pred_counts[p] += 1
            print(f"Validation predictions per class: {dict(zip(self.class_names, val_pred_counts))}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_alzheimer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {stage1_epochs + epoch + 1}")
                    break
            
            print(f"Epoch {stage1_epochs + epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)
        
        # Load best model
        self.model.load_state_dict(torch.load('best_alzheimer_model.pth'))
        
        return self.history
    
    def evaluate_model(self, test_loader):
        """
        Comprehensive model evaluation
        """
        print("\n=== Model Evaluation ===")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Alzheimer\'s Disease Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracy):
            print(f"{self.class_names[i]} Accuracy: {acc:.4f}")
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
        
        return y_pred, y_true
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if not self.history['train_loss']:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history['train_acc'], label='Training Accuracy', color='blue')
        axes[0].plot(self.history['val_acc'], label='Validation Accuracy', color='red')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history['train_loss'], label='Training Loss', color='blue')
        axes[1].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_complete(self, model_path='alzheimer_classifier_complete.pth'):
        """
        Save the complete model with preprocessing
        """
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': type(self.model).__name__,
            'class_names': self.class_names,
            'class_mapping': self.class_mapping,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'device': str(self.device)
        }, model_path)
        
        # Save metadata
        metadata = {
            'class_names': self.class_names,
            'class_mapping': self.class_mapping,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'training_date': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print("Metadata saved to model_metadata.json")

# Main training script
def main():
    """
    Main training function
    """
    print("=== Alzheimer's Disease Classification Training (PyTorch) ===\n")
    
    # Initialize classifier
    classifier = AlzheimerClassifier(img_size=224, batch_size=64)
    
    # Load data from train/test folders
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    
    # Load data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = classifier.load_data_from_folders(train_dir, test_dir)
    
    print(f"Class distribution in training: {np.bincount(train_labels)}")
    print(f"Class distribution in validation: {np.bincount(val_labels)}")
    print(f"Class distribution in test: {np.bincount(test_labels)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = classifier.create_data_loaders(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    )
    
    # Build model - CHOOSE YOUR MODEL HERE
    print("\n" + "="*50)
    print("CHOOSING MODEL: RESNET50 (Most stable and widely used)")
    print("Other options: 'efficientnet_b3', 'inception_v3'")
    print("="*50)
    model = classifier.build_model('resnet50')  # Change this to try different models
    
    # Train model
    history = classifier.train_model(train_loader, val_loader, epochs=30, learning_rate=0.002)
    
    # Evaluate model
    y_pred, y_true = classifier.evaluate_model(test_loader)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save complete model
    classifier.save_model_complete()
    
    print("\n=== Training Complete! ===")
    print("Files saved:")
    print("- best_alzheimer_model.pth (best model during training)")
    print("- alzheimer_classifier_complete.pth (final model)")
    print("- model_metadata.json (model configuration)")
    print("- confusion_matrix.png (evaluation results)")
    print("- training_history.png (training plots)")

if __name__ == "__main__":
    main() 