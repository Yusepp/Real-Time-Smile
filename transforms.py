import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import cv2
import torch

class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, *args, **kwargs):
        if type(imgs) is list:
            imgs = [self.transforms(image=np.array(img))['image'].unsqueeze(0) for img in imgs]
            tf_imgs = torch.Tensor(len(imgs), 3, 224, 224)
            torch.cat(imgs, out=tf_imgs)
            return tf_imgs
        
        return self.transforms(image=np.array(imgs))['image']


def get_transforms(augmentation=True):
    tfs_train = Transforms(A.Compose([
                                    A.Downscale(scale_max=0.5, scale_min=0.25,interpolation=cv2.INTER_AREA, p=0.5),
                                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                    A.RandomBrightnessContrast(p=0.5),
                                    A.GaussianBlur(p=0.5),
                                    A.Affine(shear={'x': 20}, p=0.5),
                                    A.Resize(224, 224),
                                    ToTensorV2()
                                ]))

    tfs_val = Transforms(A.Compose([A.Resize(224, 224),
                                    ToTensorV2()
                                ]))
    if augmentation:
        return tfs_train, tfs_val
    else:
        return tfs_val, tfs_val


def blur():
    return Transforms(A.Compose([A.GaussianBlur(p=1.0), A.Resize(224, 224), ToTensorV2()]))

def brightness():
    return Transforms(A.Compose([A.RandomBrightnessContrast(p=1.0), A.Resize(224, 224), ToTensorV2()]))

def downscale():
    return Transforms(A.Compose([A.Downscale(scale_max=0.5, scale_min=0.5,interpolation=cv2.INTER_AREA, p=1.0), A.Resize(224, 224), ToTensorV2()]))

def affine(shear={'x': 20}):
    return Transforms(A.Compose([A.Affine(shear=shear, p=1.0), A.Resize(224, 224), ToTensorV2()]))