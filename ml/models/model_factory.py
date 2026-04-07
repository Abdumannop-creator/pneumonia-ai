import torch.nn as nn
from torchvision import models

def create_model(name: str, num_classes: int = 2):
    name = name.lower()

    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
        return m

    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unknown model: {name}")
