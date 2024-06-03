'''transformer.py'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def swish(x):
    return x * torch.sigmoid(x)


def shape_list(A):
    return list(A.shape)


def split_heads(A, n_heads):
    return A.reshape(shape_list(A)[:-1] +
                     [shape_list(A)[-1] // n_heads, n_heads])


def merge_heads(A):
    return A.reshape(shape_list(A)[:-2] +
                     [shape_list(A)[-2] * shape_list(A)[-1]])


def layer_norm(x, L):
    # x = (batch, seq, data)
    mean = (torch.einsum('bij->b', x) /
            (L.reshape(-1) * x.shape[2])).reshape(-1, 1, 1)
    std_dev = ((torch.einsum('bij->b', (x - mean)**2)**(1 / 2)) /
               (L.reshape(-1) * x.shape[2])).reshape(-1, 1, 1)
    x = (x - mean) / std_dev
    return x


class Attn(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super(Attn, self).__init__()
        assert d_model % n_heads == 0,\
            "d_model must be divisible by number of heads"
        self.Wq = nn.Parameter(torch.randn(d_model, d_model).to(device) / 100)
        self.Bq = nn.Parameter(torch.randn(1, d_model).to(device) / 100)
        self.Wk = nn.Parameter(torch.randn(d_model, d_model).to(device) / 100)
        self.Wv = nn.Parameter(torch.randn(d_model, d_model).to(device) / 100)
        self.n_heads = n_heads

    def forward(self, x):
        Q, K, V = torch.matmul(
            x, self.Wq) + self.Bq, torch.matmul(x, self.Wk), torch.matmul(x, self.Wv)
        Q, K, V = split_heads(
            Q, self.n_heads), split_heads(
            K, self.n_heads), split_heads(
            V, self.n_heads)
        x = torch.einsum('bidh,bjdh->bijh', Q, K) / np.sqrt(x.shape[-1])
        x = F.softmax(x, dim=2)
        x = torch.einsum('bijh,bjdh->bidh', x, V)
        return merge_heads(x)


class AttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, device, training=True):
        super(AttnBlock, self).__init__()
        self.attn = Attn(d_model=d_model, n_heads=n_heads, device=device)
        self.W1 = nn.Parameter(torch.randn(d_model, d_model).to(device) / 100)
        self.W2 = nn.Parameter(torch.randn(d_model, d_model).to(device) / 100)

    def forward(self, x, L):
        x = self.attn(x) + x
        x = layer_norm(x, L)
        x = torch.matmul(swish(torch.matmul(x, self.W1)), self.W2) + x
        x = layer_norm(x, L)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=30, n_output=4, n_heads=3, p=.1, device=device):
        super(Transformer, self).__init__()
        self.seq = nn.ModuleList(
            [AttnBlock(d_model, n_heads, device) for i in range(10)])
        self.extract_1 = nn.Parameter(torch.randn(d_model, 200))
        self.extract_2 = nn.Parameter(torch.randn(200, n_output))
        self.p = .1

    def forward(self, x, L, training=True):
        for i in range(len(self.seq)):
            x = self.seq[i](x, L)
            if training:
                x = F.dropout(x, p=self.p)
        x = torch.einsum('...bjk->...bk', x) / (L.reshape(-1, 1) * x.shape[1])
        x = swish(torch.matmul(x, self.extract_1))
        x = torch.matmul(x, self.extract_2)
        return x
