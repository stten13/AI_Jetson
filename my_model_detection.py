import os
import torch
import torchvision
from torchvision import models
from PIL import Image
from torchvision import transforms

# Define the dataset path and classes
dataset_path = 'images' # FIX ME

# Load classes from the dataset annotations
classes = ['__background__', 'Apple', 'Orange', 'Mixed', 'Banana']
num_classes = len(classes)

# Function to load the trained model
def load_trained_model(model_path, num_classes):
    # Initialize the model with the ResNet50 backbone
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None)  # Updated to new weights API
    
    # Replace the head with the appropriate number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Load the trained weights
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle potential mismatch in class numbers
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Example usage
model_path = "fasterrcnn_fruits.pth" # FIX ME # Replace with your model's file path
trained_model = load_trained_model(model_path, num_classes)

# Function to perform inference
def predict(image, model, device):
    model.to(device)
    image = [image.to(device)]
    model.eval()
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Example for visualizing the prediction
def visualize_prediction(image, prediction, threshold=0.4):
    from torchvision.utils import draw_bounding_boxes
    import matplotlib.pyplot as plt

    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']

    # Filter out low score boxes
    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # Draw bounding boxes with labels and scores
    class_names = [classes[i] for i in labels]
    text = [(f"{name}: {score:.2f}") for name, score in zip(class_names, scores)]  # Combine label and score
    drawn_image = draw_bounding_boxes((image*255).byte(), boxes, text, width=4)  # Fixed image scaling
    
    plt.figure(figsize=(12, 8))
    plt.imshow(drawn_image.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()

# Assuming 'device' is 'cpu' or 'cuda'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Replace with your image path
image_path = 'images/8075c2f0-orange1.jpeg' # FIX ME # Replace with your image path

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

# Load and preprocess the image
image = preprocess_image(image_path)

# Perform inference
prediction = predict(image, trained_model, device)

# Visualize the prediction
visualize_prediction(image, prediction)