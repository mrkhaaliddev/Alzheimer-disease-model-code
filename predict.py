import argparse
import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

# Add the directory containing alzheimer_classifier_training.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import AlzheimerClassifier for architecture and loading weights
from alzheimer_classifier_training import AlzheimerClassifier

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use cv2 for image loading and color conversion
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found or cannot be read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for torchvision transforms
    image_pil = Image.fromarray(image)

    return transform(image_pil).unsqueeze(0)

# Prediction function
def make_prediction(model, image_path):
    # Preprocess the image
    input_tensor = preprocess_image(image_path)
    
    # Move tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_index].item() * 100
    
    # Map class index to label (Alzheimer's disease classes)
    class_mapping = {
        0: 'No Impairment', 
        1: 'Very Mild Impairment', 
        2: 'Mild Impairment', 
        3: 'Moderate Impairment'
    }
    predicted_label = class_mapping.get(predicted_class_index, 'UNKNOWN')
    
    return predicted_label, confidence

# Model weights paths - Updated to match your actual model files
MODEL_PATHS = {
    'best': 'models/Resnet_50_alzheimer_classifier_90%_.pth',
    'complete': 'models/old_best_alzheimer_model_resnet_50.pth',
    'second': 'best_alzheimer_model.pth'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Alzheimer\'s disease stage from brain MRI image')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='best', choices=['best', 'complete', 'second'], 
                       help='Which model to use for prediction (best, complete, or second)')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
        
    # Initialize the classifier and build the model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    classifier = AlzheimerClassifier(device=device)
    classifier.build_model('resnet50')  # Use the same model architecture as training

    # Get the appropriate model path
    model_weights_path = MODEL_PATHS[args.model]
    
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at {model_weights_path}")
        print("Available models:")
        for model_name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                print(f"  ✓ {model_name}: {path}")
            else:
                print(f"  ✗ {model_name}: {path} (not found)")
        sys.exit(1)

    # Load the trained weights into the model's state_dict
    try:
        print(f"Loading model from: {model_weights_path}")
        print(f"File exists: {os.path.exists(model_weights_path)}")
        print(f"File size: {os.path.getsize(model_weights_path) / (1024*1024):.2f} MB")
        
        # Load the model weights
        checkpoint = torch.load(model_weights_path, map_location=device)
        
        # Check if it's a complete model (with metadata) or just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading complete model with metadata...")
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading state dict directly...")
            classifier.model.load_state_dict(checkpoint)
            
        classifier.model.eval()
        print(f"✅ Model loaded successfully from {model_weights_path}")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Make prediction
    prediction, confidence = make_prediction(classifier.model, args.image)
    
    # Print prediction
    print(f"\n🎯 Prediction Results:")
    print(f"📁 Image: {args.image}")
    print(f"🧠 Predicted Stage: {prediction}")
    print(f"📊 Confidence: {confidence:.2f}%")
    
    # Additional interpretation
    if confidence < 50:
        print("⚠️  Low confidence prediction - consider manual review")
    elif confidence > 80:
        print("✅ High confidence prediction")
    else:
        print("⚠️  Moderate confidence prediction")
    
    print(f"\n📋 Model Info:")
    print(f"   - Architecture: ResNet50")
    print(f"   - Classes: 4 Alzheimer's stages")
    print(f"   - Image Size: 224x224")
    print(f"   - Device: {device}") 