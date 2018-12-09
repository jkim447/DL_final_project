import torch
import torchvision
from torchvision import models

m = models.vgg16(pretrained=True)
print(m)
