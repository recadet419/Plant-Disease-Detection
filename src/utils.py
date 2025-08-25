import torch
import torch.nn as nn
from torchvision import transforms
import os
import torch.nn.functional as F
from torchvision.transforms import Lambda
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
from timm import create_model


def get_transforms_for_model(augment):
    if augment:
        # data_transforms = transforms.Compose(  # trying to reproduce the same results in the paper
        #     [
        #         transforms.Resize((224, 224)),  # Resize images to 224x224
        #         transforms.RandomHorizontalFlip(),  # Horizontal flipping
        #         transforms.RandomVerticalFlip(),  # Vertical flipping
        #         transforms.RandomRotation(15),  # Rotation
        #         transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # Zooming
        #         transforms.ColorJitter(brightness=0.2),  # Brightness adjustment
        #         transforms.RandomAffine(
        #             0, translate=(0.1, 0.1)
        #         ),  # Horizontal and vertical shifting
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        #         ),  # Normalization
        #     ]
        # )
        data_transforms = transforms.Compose(  # My settings
            [
                transforms.Resize(256),  
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.95, 1.05),
                    ratio=(0.75, 1.33),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation([-40, 40]),
                transforms.ColorJitter(saturation=0.8, hue=0.021),
                # Add contrast and saturation adjustment
                transforms.RandomAffine(
                    degrees=0,
                    translate=(
                        0.13,
                        0.13,
                    ),  # Increase translate values for more shifting
                    scale=(
                        0.8,
                        1.2,
                    ),  # Adjust scale values for additional zooming effect
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return data_transforms


# Get the number of classes to feed in the model architecture as FEATURES
DATA = "data"
CLASSES = sorted(os.listdir(DATA))
FEATURES = len(CLASSES)



# ---------- Utility: build kNN adjacency (cosine), add self-loops, row-normalize ----------
def knn_adjacency(features, k=8, eps=1e-6):
    """
    features: (N, C) L2/row-normalized inside
    returns A_hat: (N, N) dense row-normalized adjacency with self-loops
    """
    f = F.normalize(features, p=2, dim=1)              
    sim = f @ f.t()                                   
    # remove self for kNN selection, add self-loop later
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9

    # top-k neighbors per node
    idx = torch.topk(sim, k=k, dim=1).indices         

    N = sim.size(0)
    A = torch.zeros((N, N), device=features.device)
    A.scatter_(1, idx, 1.0)                           
    # symmetrize
    A = torch.maximum(A, A.t())
    # add self loops
    A = A + torch.eye(N, device=A.device)

    # row-normalize A
    deg = A.sum(dim=1, keepdim=True) + eps
    A_hat = A / deg
    return A_hat