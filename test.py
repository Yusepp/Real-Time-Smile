import cv2
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from PIL import Image
from time import time
from model import SmileDetector
from transforms import get_transforms
from utils_smile import detect_face

cudnn.benchmark = True

# Read the arguments
parser = argparse.ArgumentParser(description='Run Inference RTSD for Smile Detection')
parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',default='RTSD_MNetv3.pth')

default = [[f, f.replace('.jpg', '_light.jpg'), f.replace('.jpg', '_blur.jpg'), f.replace('.jpg', '_low.jpg')] for f in ['test_images/inputs/test_input_1.jpg','test_images/inputs/test_input_2.jpg','test_images/inputs/test_input_3.jpg',
                             'test_images/inputs/test_input_4.jpg']]

parser.add_argument('-i', '--input', nargs='+', help='Sample input image path', default=sum(default, []))

default = [[f, f.replace('.jpg', '_light.jpg'), f.replace('.jpg', '_blur.jpg'), f.replace('.jpg', '_low.jpg')] for f in ['test_images/outputs/test_output_1.jpg','test_images/outputs/test_output_2.jpg','test_images/outputs/test_output_3.jpg',
                             'test_images/outputs/test_output_4.jpg']]

parser.add_argument('-o', '--output', nargs='+', help='Sample output image path', default=sum(default, []))

args = parser.parse_args()
variables = vars(args)

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SmileDetector(freeze=False)
model.load_state_dict(torch.load(variables['weights']))
model.to(device)
model.eval()

# Read the images
imgs = [Image.open(f) for f in variables['input']]
tfs, _ = get_transforms(augmentation=False)

# Run inference
with torch.no_grad():
    for i, img in enumerate(imgs):
        crops, boxes = detect_face(img)
        
        # Preprocess images
        start = time()
        crops = tfs(crops).to(device)
        end = time()
        print(f'Preprocess time {int((end - start)*1000)}ms ({int((end - start)*1000)//len(crops)}ms/image)')
        
        # Infer Model
        start = time()
        smile_scores = model(crops)
        end = time()
        print(f'Inference time {int((end - start)*1000)}ms ({int((end - start)*1000)//len(crops)}ms/image)')
        
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for (left, top, right, bottom), score in zip(boxes, smile_scores):
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 255), 2)
            if img.shape[1] < 1025:
                cv2.putText(img, f'Smile Score: {score[0]:.4f}',(int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)
            else:
                cv2.putText(img, f'Smile Score: {score[0]:.4f}',(int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
        
        cv2.imwrite(variables['output'][i], img)