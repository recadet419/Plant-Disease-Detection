import os
import torch
import torch.optim as optim
import torch.nn as nn
from models import (ViT, EfficientNetV2B3, EfficientNetV2B3ViT, MobileNetV3_large, VGG16, ResNet50, DenseNet121,
                    MobileNetV3ViT, VGG16ViT, ResNet50ViT, DenseNet121ViT, ConvNeXt, ConvNeXtViT, ConvNeXtGNN, EfficientNetGNN
                    )
from time import gmtime, strftime
from dotenv import load_dotenv
import wandb
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
load_dotenv()
time = strftime("%d- %H:%M:%S", gmtime())

# Dataset configurations
DATA = "data"  
TEST_SIZE = 0.1
VALI_SIZE = 0.1
RANDOM_STATE = 42 
BATCH_SIZE = 64
CLASSES = sorted(os.listdir(DATA))

# Training configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()
EPOCHS = 70
lr = 0.0001

# Switch between training and testing, augmenting the dataset, and the type of dataset
TRAINING = False
AUGMENT = True  
DATATYPE = "potatodata" 

NEW_DATASET = True  

if TRAINING:
    MODELS = {
        # "EffNetViT": EfficientNetV2B3ViT().to(DEVICE),
        # "ConvNeXt": ConvNeXt().to(DEVICE),
        # "ConvNeXtViT": ConvNeXt().to(DEVICE),
        # "ConvNeXtGNN": ConvNeXtGNN().to(DEVICE),
        "EfficientNetGNN": EfficientNetGNN().to(DEVICE),
    }

    OPTIMIZERS = {
        # "EffNetViT": optim.Adam(MODELS["EffNetViT"].parameters(), lr, weight_decay=0.0001),
        # "ConvNeXt": optim.Adam(MODELS["ConvNeXt"].parameters(), lr, weight_decay=0.0001),
        # "ConvNeXtViT": optim.Adam(MODELS["ConvNeXtViT"].parameters(), lr, weight_decay=0.0001),
        # "ConvNeXtGNN": optim.Adam(MODELS["ConvNeXtGNN"].parameters(), lr, weight_decay=0.0001),
        "EfficientNetGNN": optim.Adam(MODELS["EfficientNetGNN"].parameters(), lr, weight_decay=0.0001),
    }
    SCHEDULER = {
        # "EffNetViT": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["EffNetViT"], patience=5, factor=0.5, verbose=True
        # ),
        # "ConvNeXt": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["ConvNeXt"], patience=5, factor=0.5, verbose=True
        # ),
        #   "ConvNeXtViT": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["ConvNeXtViT"], patience=5, factor=0.5, verbose=True
        # ),
        #   "ConvNeXtGNN": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["ConvNeXtGNN"], patience=5, factor=0.5, verbose=True
        # ),
        "EfficientNetGNN": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["EfficientNetGNN"], patience=5, factor=0.5, verbose=True
        ),
    }


else:  # Testing
    MODELS = {
        "EfficientNetGNN": EfficientNetGNN,
    }

SAVED_MODELS = {}
if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            # "EffNetViT": "EffNetViT_potatodata_Aug_True_174940_2025.pth",
            #   "ConvNeXt": "ConvNeXt_potatodata_Aug_True_183543_2025.pth",
            # "ConvNeXtGNN": "ConvNeXtGNN_potatodata_Aug_True_041414_2025.pth",
            # "ConvNeXtGNN": "ConvNeXtGNN_potatodata_AMobileug_True_014437_CNNs.pth",
            # "EfficientNetGNN": "EfficientNetGNN_potatodata_AMobileug_True_014437_CNNs.pth",
            "EfficientNetGNN": "EfficientNetGNN_potatodata_Aug_True_081613_2025.pth",
        }
    else:
        # For NEW_DATASET and not AUGMENT
        SAVED_MODELS = {}
else:
    # Not NEW_DATASET
    if AUGMENT:
        SAVED_MODELS = {
            # "EffNetViT": "EfficientNetV2B3_last_plantVillage_Aug_True_151547_HT400k.pth",
            # "ConvNeXtViT": "ConvNeXtViT_last_plantVillage_Aug_151547_HT400k.pth",
            # "ConvNeXtGNN": "ConvNeXtGNN_last_plantVillage_Aug_151547_HT400k.pth",
            "EfficientNetGNN": "EfficientNetGNN_last_plantVillage_Aug_151547_HT400k.pth",
        }
    else:
        # Not NEW_DATASET and not AUGMENT
        SAVED_MODELS = {}


wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    name=f"{time}_{DATATYPE}_train_Aug_{AUGMENT}",  
)
