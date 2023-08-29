from torch import nn
import torch
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_funcs import *


device = torch.device('mps' if torch.has_mps else 'cpu')

n_samples = 1000

epochs = 1500

X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=82)

X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle= True, random_state=81)


class BinaryCircleClasserV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=16, device=device)
        self.layer2 = nn.Linear(in_features=16, out_features=8, device=device)
        self.layer3 = nn.Linear(in_features=8, out_features=1, device=device)
        self.elu = nn.ELU()

    def forward(self, x):
        return self.layer3(self.elu(self.layer2(self.elu(self.layer1(x)))))

torch.manual_seed(81)
model_0 = BinaryCircleClasserV0().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model_0.parameters(), lr=0.3)

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(X_train)
    y_preds = y_logits.sigmoid().round()

    loss = criterion(y_logits, torch.unsqueeze(y_train, dim=1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        model_0.eval()
        with torch.inference_mode():
            y_preds = model_0(X_test)
            test_loss = criterion(y_preds, y_test.unsqueeze(1))
            accuracy = accuracy_fn(y_test.unsqueeze(1), torch.sigmoid(y_preds).round())
        print(f"Epoch: {epoch} | Accuracy: {accuracy:.2f}% | Loss: {loss.data:.6f} | Test Loss: {test_loss.data:.6f}")

plot_decision_boundary(model_0, X_test, y_test)
plt.show()