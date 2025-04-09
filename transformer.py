import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, output_size = 384,num_heads=8, dropout = 0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size,num_heads,dropout=dropout,batch_first=True)

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

        self.ffn = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size,hidden_size),
        )



    def forward(self, x):
        attn_output,_=self.self_attn(x,x,x)
        x = x + attn_output
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x

class Transformer(nn.Module):
    def __init__(self,depth=6):
        super(Transformer,self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock() for _ in range(depth)
        ])

        self.out_layer = nn.Sequential(
            nn.Linear(1024,384),
            nn.ReLU(),
            nn.Linear(384,384)
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x)
