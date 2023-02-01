import wandb
import os
import argparse
import torch
import torch.backends.cudnn as cudnn

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

from model import SmileDetector
from transforms import get_transforms

#torch.set_float32_matmul_precision(precision)

def parse_variables():
    parser = argparse.ArgumentParser(description='Train MobileNetV3 based real time smile detector')
    parser.add_argument('-b', '--batch', type=int, help='Batch Size', default=256)
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs', default=100)
    parser.add_argument('-w', '--workers', type=int, help='Workers for datalaoder', default=2)
    parser.add_argument('-p', '--patience', type=int, help='Number of patience for early stopping', default=0)
    parser.add_argument('-d', '--dataset', type=str, help='Root path to dataset', default='dataset')
    parser.add_argument('-m', '--mode', type=str, help='dev = train; prod = train + val', default='dev')
    
    parser.add_argument('--freeze', action='store_true', help='Finetune')
    parser.set_defaults(freeze=False)
    
    parser.add_argument('--logging', action='store_true', help='Log info wandb')
    parser.set_defaults(logging=False)
    
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.set_defaults(augment=False)

    
    args = parser.parse_args()
    variables = vars(args)
    return variables

# Read Args
variables = parse_variables()

# Enable cudnn kernels
cudnn.benchmark = True
BATCH_SIZE = variables['batch']

# Load transformations
tfs_train, tfs_val = get_transforms(variables['augment'])

# Load ImageFolder
train_ds = ImageFolder(root=os.path.join(variables['dataset'],'train'), transform=tfs_train)
val_ds = ImageFolder(root=os.path.join(variables['dataset'],'val'), transform=tfs_val)
test_ds = ImageFolder(root=os.path.join(variables['dataset'],'test'), transform=tfs_val)

# DataLoaders
if variables['mode'] == 'dev':
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=variables['workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=variables['workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=variables['workers'], pin_memory=True)
    
elif variables['mode'] == 'prod':
    train_loader = DataLoader(ConcatDataset([train_ds, val_ds]), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Load Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SmileDetector(variables['freeze']).to(device)

# Train and Test
if variables['mode'] == 'dev':
    model.fit(train_loader, batch_size=BATCH_SIZE, epochs=variables['epochs'], val_data=val_loader, patience=variables['patience'])
    
elif variables['mode'] == 'prod':
    model.fit(train_loader, batch_size=BATCH_SIZE, epochs=variables['epochs'], val_data=test_loader, patience=variables['patience'])
    
model.evaluate(test_loader, batch_size=BATCH_SIZE)

# Finishing wandb process
if variables['logging']:
    wandb.finish()
    
# Saving whole model
torch.save(model, 'best_smile.pt')
print('Model Saved!')