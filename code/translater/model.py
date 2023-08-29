import torch
from torch import nn
import math
import numpy as np
from PositionalEncoder import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self,
                num_tokens,
                dim_model,
                num_heads,
                num_encoder_layers,
                num_decoder_layers,
                dropout_percent) -> None:
        super().__init__()
        
        self.model_type = "Transformer"
        self.dim_model = dim_model
        
        self.positional_encoder = PositionalEncoding(dim_model=dim_model,
            dropout_percent=dropout_percent,
            max_len= 5000)
        
        self.embedding = nn.Embedding()
        
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers= num_encoder_layers,
            num_decoder_layers= num_decoder_layers,
            dropout= dropout_percent
        )
        self.out = nn.Linear(dim_model, num_tokens)
    
    def forward(self, src, tgt, tgt_mask = None, src_pad_mask = None, tgt_pad_mask = None):
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        src = tgt.permute(1, 0, 2)
        tgt = src.permute(1, 0, 2)
        
        return self.out(self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask))
    def get_tgt_mask(self, size) -> torch.tensor:
        
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int):
        return (matrix == pad_token)