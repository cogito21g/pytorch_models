import torch.nn as nn
import torchvision.models as models

def get_alexnet_model():
    model = models.alexnet(pretrained=False)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 첫 번째 레이어 수정
    model.features[2] = nn.MaxPool2d(kernel_size=2, stride=2)  # 두 번째 풀링 레이어 수정
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # CIFAR-10은 10개의 클래스를 가집니다.
    return model