import argparse
import torch
import torch.backends.cudnn as cudnn

from albumentations.pytorch import ToTensorV2
from PIL import Image
from model import SmileDetector
from transforms import get_transforms


# Read the arguments
parser = argparse.ArgumentParser(description='Run Inference RTSD for Smile Detection')
parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                    default='best_smile.pt')

parser.add_argument('-i', '--input', nargs='+', help='Sample input image path',
                    default=['test_images/test_input.jpg','test_images/test_input_2.jpg','test_images/test_input_3.jpg',
                             'test_images/test_input_4.jpg','test_images/test_input_5.jpg'])

args = parser.parse_args()
variables = vars(args)

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(variables['weights'])                
model.to(device)
model.eval()

# Read the images
imgs = [Image.open(f) for f in variables['input']]
tfs, _ = get_transforms(augmentation=False)

# Run inference
with torch.no_grad():
    for i, img in enumerate(imgs):
        img = tfs(img).to(device).to(torch.float32).unsqueeze(0)
        smile_score = model(img)[0][0]
        print(f"{variables['input'][i]} smile score: {smile_score:.4f}")