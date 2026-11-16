"""Small CNN for the Essentials of Deep Learning project."""
import torch
from torch import nn
class LungCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(64*8*8,64),nn.ReLU(),nn.Dropout(.35),nn.Linear(64,3))
    def forward(self,x):return self.classifier(self.features(x))
