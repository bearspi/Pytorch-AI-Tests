import torch
from tqdm import tqdm
from torchmetrics import Accuracy

def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, EPOCHS: int, device: torch.device):
    for epoch in tqdm(range(EPOCHS)):
        avrg_loss, avrg_acc = 0, 0
        for batch, (X, y) in enumerate(dataloader):
            model, X, y = model.to(device), X.to(device), y.to(device)
            
            model.train()
            
            y_predictions = model(X)
            
            loss = criterion(y_predictions, y)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            accuracy = Accuracy(task="multiclass", num_classes=len(dataloader.dataset.classes)).to(device)
            
            acc = accuracy(y_predictions, y)
            
            avrg_loss += loss
            
            avrg_acc += acc * 100
            
            if batch % 400 == 0:
                print(f"\n Looked at {batch * len(X)}/{len(dataloader.dataset)} samples")
                
        avrg_loss /= len(dataloader)
        avrg_acc /= len(dataloader)
        
        print(f"\nTraining Epoch: {epoch + 1} \n Loss: {avrg_loss:.6} | Accuracy: {avrg_acc:.4}% \n")
        

def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device):
    model = model.to(device)
    model.eval()
    avrg_loss, avrg_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device=device), y.to(device=device,)
            
            y_predictions = model(X)
            
            loss = criterion(y_predictions, y)
            
            accuracy = Accuracy(task="multiclass", num_classes=len(dataloader.dataset.classes)).to(device)
            
            acc = accuracy(y_predictions, y)
            
            avrg_loss += loss
            
            avrg_acc += acc * 100
        avrg_loss /= len(dataloader)
        avrg_acc /= len(dataloader)
        print(f"\nTesting Results \n Loss: {avrg_loss:.6} | Accuracy: {avrg_acc:.4}% \n")
        return {"model_accuracy": avrg_acc, "model_loss": avrg_loss}
    
def make_predictions(model: torch.nn.Module, data: list, device: torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)