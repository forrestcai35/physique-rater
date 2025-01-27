import torch.nn as nn
from torchvision import models

def build_resnet_multioutput(num_outputs=11):
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Modify final layers
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_outputs)
    )
    return model
