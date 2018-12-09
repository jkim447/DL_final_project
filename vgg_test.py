import torch
import torchvision
from torchvision import models

m = models.resnet18(pretrained=True)
total = sum(p.numel() for p in m.parameters())
print('total number of params: ', total)
total = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('total trainalbe params:', total)
