import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10은 10개의 클래스를 가집니다.
    return model