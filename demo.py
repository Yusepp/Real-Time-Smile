import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from time import time, sleep
from transforms import get_transforms
from model import SmileDetector
from utils import detect_face, get_intensity_plot_array

# Read the arguments
parser = argparse.ArgumentParser(description='Run Inference RTSD for Smile Detection')
parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                    default='RTSD_MNetv3.pth')
args = parser.parse_args()
variables = vars(args)
total = []
scores = []
net = variables["weights"].replace(".pth", "")

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else device
model = SmileDetector(freeze=False, net=net)
model.to(device)
model.load_state_dict(torch.load(variables['weights'], map_location=torch.device(device)))
model.eval()

# Get transformations
tfs, _ = get_transforms(augmentation=False)

# Read webcam loop
cap = cv2.VideoCapture(0)
first = True
while cap.isOpened():
    success, image = cap.read()
    if first:
        for i in range(10):
            success, image = cap.read()
        first = False
    # Format image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect Faces
    crops, boxes = detect_face(img)    

    if len(crops) > 0:
        # Preprocess
        start_time = time()
        crops = tfs(crops).to(device)
        
        # Inference
        smile_scores = model(crops).cpu().detach().numpy()
        
        # Save fps
        fps = 1/(time() - start_time)
        total.append(fps)
        
        
        
        # Detections    
        for (left, top, right, bottom), score in zip(boxes, smile_scores):
            scores.append(score[0])
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            cv2.putText(image, f'Smile Score: {score[0]:.4f}',(int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        
        # Print stats
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2)
        cv2.putText(image, f'Avg. FPS: {np.mean(total):.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2)
        cv2.putText(image, f'Max. FPS: {max(total):.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2)
        cv2.putText(image, f'Min. FPS: {min(total):.2f}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2)
        
        intensities = get_intensity_plot_array(scores)
        intensities = cv2.resize(src=intensities, dsize=(image.shape[1], image.shape[0]))

        
        image = cv2.vconcat([image, intensities])
    
    # Show image
    cv2.imshow('RTSD: Real-Time Smile Detector', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()
    