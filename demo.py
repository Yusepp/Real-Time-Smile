import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from time import time
from transforms import get_transforms
from model import SmileDetector
from utils import detect_face

# Read the arguments
parser = argparse.ArgumentParser(description='Run Inference RTSD for Smile Detection')
parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                    default='RTSD_MNetv3.pth')
args = parser.parse_args()
variables = vars(args)
total = []

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SmileDetector(freeze=False)
model.load_state_dict(torch.load(variables['weights']))
model.to(device)
model.eval()

# Get transformations
tfs, _ = get_transforms(augmentation=False)

# Read webcam loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    # Format image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect Faces
    crops, boxes = detect_face(img)    

    # Preprocess
    start_time = time()
    crops = tfs(crops).to(device)
    
    # Inference
    smile_scores = model(crops)
    
    # Save fps
    fps = 1/(time() - start_time)
    total.append(fps)
    
    # Detections    
    for (left, top, right, bottom), score in zip(boxes, smile_scores):
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, f'Smile Score: {score[0]:.4f}',(int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Print stats
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(image, f'Avg. FPS: {np.mean(total):.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(image, f'Max. FPS: {max(total):.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(image, f'Min. FPS: {min(total):.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Show image
    cv2.imshow('RTSD: Real-Time Smile Detector', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()
    