import argparse
import torch
import torch.backends.cudnn as cudnn

from PIL import Image
from time import time
from model import SmileDetector
from transforms import get_transforms

cudnn.benchmark = True

# Read the arguments
parser = argparse.ArgumentParser(description='Run Inference RTSD for Smile Detection')
parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                    default='RTSD_MNetv3.pth')

parser.add_argument('-i', '--input', nargs='+', help='Sample input image path',
                    default=['test_images/test_input.jpg','test_images/test_input_2.jpg','test_images/test_input_3.jpg',
                             'test_images/test_input_4.jpg','test_images/test_input_5.jpg'])

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
    # Preprocess images
    start = time()
    imgs = tfs(imgs).to(device)
    end = time()
    print(f'Preprocess time {int((end - start)*1000)}ms ({int((end - start)*1000)//len(variables["input"])}ms/image)')
    
    # Infer Model
    start = time()
    smile_scores = model(imgs)
    end = time()
    print(f'Inference time {int((end - start)*1000)}ms ({int((end - start)*1000)//len(variables["input"])}ms/image)')
    
    # Print results
    for i in range(len(imgs)):
        print(f"{variables['input'][i]} smile score: {smile_scores[i][0]:.4f}")