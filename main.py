import torch
from torch import nn
import matplotlib.pyplot as plt

mps_device = torch.device('mps:0' if torch.has_mps else 'cpu')



weigth = 0.69
bias = 0.31 

start = 0
end = 1
step = 0.02

epochs = 200



X = torch.arange(start=start, end=end, step=step).unsqueeze(1)

y = weigth * X + bias

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data = X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="y", s=4, label="Testing data")
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14});
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(81)
model_0 = LinearRegressionModel()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    print(f"Loss: {loss}")

with torch.inference_mode():
    plot_predictions(predictions=model_0(X_test))






