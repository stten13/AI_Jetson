import os
import time
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Define class labels
classes = ['__background__', 'Apple', 'Banana', 'Orange']
num_classes = len(classes)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
def load_trained_model(model_path, num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage.cpu()))
    model.to(device)
    model.eval()
    return model

model_path = "fasterrcnn_fruits.pth"  # Replace with your model path
model = load_trained_model(model_path, num_classes)

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Prediction function
def predict_image(image, model, threshold=0.4):
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        outputs = model([image_tensor])[0]
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            results.append((box, score, classes[label]))
    return results

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

# FPS tracking
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Convert to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Get predictions
    results = predict_image(pil_image, model)

    # Draw boxes and labels
    for box, score, label in results:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Draw FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Display frame
    cv2.imshow('Fruit Detection', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

