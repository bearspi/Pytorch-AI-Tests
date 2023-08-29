import torch, torchvision
from torch import nn 
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import random

from helper_funcs import *
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from train_and_test_funcs import *
from pathlib import Path

device = torch.device('mps' if torch.has_mps else 'cpu')

#Modeli kaydetmek için bir dosya ayarlıyoruz
Model_Path = Path("models")
Model_Path.mkdir(parents=True, exist_ok=True)

#Modelin ismini koyuyoruz
Model_Name = "Pytorch_FMNIST_Model_6_TinyVGG.pth"
Model_Save_Path = Model_Path / Model_Name

BATCH_SIZE = 32
EPOCHS = 100

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(), target_transform=None)

test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(), target_transform=None)

train_dataLoader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

test_dataLoader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

class FMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.layer_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(output_size=14),
            nn.Dropout2d()
        )
        self.layer_conv_stack1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(output_size=7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.layer_conv_stack(x)
        x = self.layer_conv_stack1(x)
        x = self.classifier(x)
        return x

torch.manual_seed(82)

model_0 = FMNISTModelV0(input_shape=1, hidden_units=128, output_shape=len(train_data.class_to_idx))


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

train_loop(model_0, train_dataLoader, criterion, optimizer, EPOCHS, device)
test_loop(model_0, test_dataLoader, criterion, device)
torch.save(model_0.state_dict(), Model_Save_Path)