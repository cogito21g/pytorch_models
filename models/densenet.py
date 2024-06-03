import torch.nn as nn
import torchvision.models as models

def get_densenet_model():
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 10)  # CIFAR-10은 10개의 클래스를 가집니다.
    return model