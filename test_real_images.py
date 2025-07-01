import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from alzheimer_classifier_training import AlzheimerClassifier

def test_single_image(model_path, image_path):
    """Test a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['No Impairment', 'Very Mild Impairment', 'Mild Impairment', 'Moderate Impairment']
    
    # Load model
    classifier = AlzheimerClassifier(device=device)
    classifier.build_model('resnet50')
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            classifier.model.load_state_dict(checkpoint)
        classifier.model.eval()
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load and preprocess image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply validation transforms
        transform = classifier.get_val_transforms()
        transformed = transform(image=image)
        tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = classifier.model(tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
            
            # Get all probabilities
            all_probs = probabilities[0].cpu().numpy()
        
        print(f"\n🎯 Prediction Results:")
        print(f"📁 Image: {image_path}")
        print(f"🧠 Predicted Stage: {class_names[predicted_class]}")
        print(f"📊 Confidence: {confidence:.2f}%")
        
        print(f"\n📋 All Class Probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, all_probs)):
            print(f"   {class_name}: {prob*100:.2f}%")
        
        if confidence < 50:
            print("\n⚠️  Low confidence prediction - consider manual review")
        elif confidence > 80:
            print("\n✅ High confidence prediction")
        else:
            print("\n⚠️  Moderate confidence prediction")
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': all_probs
        }
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def test_folder(model_path, folder_path, expected_label=None):
    """Test all images in a folder"""
    print(f"Testing images from: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images to test")
    
    results = []
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        print(f"\n--- Image {i+1}/{len(image_files)}: {image_file} ---")
        
        result = test_single_image(model_path, image_path)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n📊 Summary:")
        print(f"   Total images tested: {len(results)}")
        
        # Count predictions per class
        pred_counts = {}
        for result in results:
            label = result['predicted_label']
            pred_counts[label] = pred_counts.get(label, 0) + 1
        
        print(f"   Predictions per class:")
        for label, count in pred_counts.items():
            print(f"     {label}: {count}")
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"   Average confidence: {avg_confidence:.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Alzheimer classifier with real images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.model, args.image)
    elif args.folder:
        test_folder(args.model, args.folder)
    else:
        print("Please provide either --image or --folder argument") 