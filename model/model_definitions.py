import torch
import torch.nn as nn
from torchvision import models

def build_resnet_multioutput(num_outputs=7):
    """
    Builds a ResNet50 pre-trained on ImageNet, modifies final layer to output
    `num_outputs` continuous values (for multi-output regression).
    """
    model = models.resnet50(pretrained=True)
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_outputs)

    return model
