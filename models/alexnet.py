import torch.nn as nn
import torchvision.models as models

def get_alexnet_model():
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # CIFAR-10은 10개의 클래스를 가집니다.
    return model
