import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_percent, max_len) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_percent)
        
        positional_enc = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        
        positional_enc[:, 0::2] = torch.sin(positions_list * division_term)
        positional_enc[:, 1::2] = torch.sin(positions_list * division_term)
        
        positional_enc = positional_enc.unsqueeze(0, 1).transpose(0, 1)
        self.register_buffer("positional_enc", positional_enc)
        
    def forward(self,
                token_embedding: torch.tensor) -> torch.tensor: 
        return self.dropout(token_embedding + self.positional_enc[:token_embedding.size(0), :])
