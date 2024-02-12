import numpy as np
import torch
import torch.nn as nn

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)
src = torch.rand((1, 10, 512))
tgt = torch.rand((1, 1, 512))

for i in range(src.shape[1]):
    out = transformer_model(src, tgt)
    tgt = torch.cat((tgt, out), dim=1)
print(tgt.shape)
