import wandb
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import freeze_support

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

from model import SmileDetector
from transforms import get_transforms, blur, downscale, brightness, affine

if __name__ == '__main__':
    freeze_support()

    # Read the arguments
    parser = argparse.ArgumentParser(description='Run Inference RTSD for Smile Detection')
    parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default='RTSD_MNetv3.pth')
    parser.add_argument('-d', '--dataset', type=str, help='Path to root dataset',
                        default='datasets/lfwcrop')
    parser.add_argument('-n', '--net', type=str, help='Select backbone', default='MNet-L')
    args = parser.parse_args()
    variables = vars(args)

    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SmileDetector(net=variables['net'], freeze=False)
    model.load_state_dict(torch.load(variables['weights']))
    model.to(device)
    model.eval()

    # Load transformations
    tfs_list = [('Normal', get_transforms(augmentation=False)[0]), ('Blur', blur()), ('Downscale', downscale()), ('Brightness', brightness()), ('Shear X', affine())]
    for name, tfs in tfs_list:
        print(name)
        # Dataset
        ds = ImageFolder(root=os.path.join(variables['dataset']), transform=tfs)
        data_loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

        # Evaluate
        model.evaluate(data_loader, batch_size=16, logging=False)