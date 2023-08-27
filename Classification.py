from torch import nn
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_funcs import *
import matplotlib.pyplot as plt


device = torch.device('mps' if torch.has_mps else 'cpu')

n_sample = 3000
n_features = 2
n_classes = 4

epochs = 3000

X, y = make_blobs(n_samples=n_sample, n_features=n_features, centers=n_classes, cluster_std=0.7, random_state=156)

X = torch.from_numpy(X).type(torch.float).to(device=device)
y = torch.from_numpy(y).type(torch.LongTensor).to(device=device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle= True, random_state=82)

class BlobClasserModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.LinearLayerStack = nn.Sequential(
            nn.Linear(in_features=2, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=16),
            nn.Linear(in_features=16, out_features=4)
        )
    def forward(self, x):
        return self.LinearLayerStack(x)

torch.manual_seed(83)
model_0 = BlobClasserModelV0().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_0.parameters(), lr=0.3)

for epoch in range(epochs):
    model_0.train()
    
    y_logits = torch.squeeze(model_0(X_train), 1)
    
    loss = criterion(y_logits, y_train)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    

    
    if epoch % 10 == 0:
        model_0.eval()
        with torch.inference_mode():
            y_preds = model_0(X_test)
            test_loss = criterion(y_preds, y_test)
            accuracy = accuracy_fn(y_test, torch.softmax(y_preds, dim=1).argmax(dim=1))
        print(f"Epoch: {epoch} | Accuracy: {accuracy:.2f}% | Loss: {loss.data:.4f} | Test Loss: {test_loss.data:.5f}")

plot_decision_boundary(model_0, X_test, y_test)
plt.show()