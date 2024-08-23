import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DREMLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads,relation_cnt=5, dropout=0.5):
        super(DREMLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.in_drop=nn.Dropout(0.2)
        self.relation_cnt=relation_cnt
        self.W = nn.ModuleList([nn.Linear(in_features, self.head_dim) for _ in range(num_heads)])
        self.W_r = nn.ModuleList([nn.ModuleList([nn.Linear(in_features, self.head_dim) for _ in range(num_heads)]) for _ in range(relation_cnt)])
        self.attn_dropout = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_linear = nn.Linear(self.head_dim * num_heads, out_features)

    def forward(self, x, adj):
        outputs = []
        x=self.in_drop(x)
        adj = torch.einsum('ijkl->ikl', adj)
        adj = torch.einsum('ijk,ijl->ijl', torch.Tensor(adj), x)
        for i in range(self.num_heads):
            h_prime = self.W[i](x)
            attn_scores_r = []
            for j in range(self.relation_cnt):
                W_r_head = self.W_r[j][i](adj)
                attn_scores = torch.matmul(h_prime, W_r_head.transpose(1, 2))
                attn_scores_r.append(attn_scores)
            attn_scores_r = torch.stack(attn_scores_r, dim=1)
            attn_scores = torch.sum(attn_scores_r, dim=1)
            attn_scores = torch.sum(attn_scores, dim=0)
            attn_scores = F.leaky_relu(attn_scores, negative_slope=0.2)
            attn_scores = F.softmax(attn_scores, dim=-1)
            attn_scores = self.attn_dropout(attn_scores)
            h_prime = torch.matmul(attn_scores, h_prime)
            outputs.append(h_prime)
        output = torch.cat(outputs, dim=-1)
        output = self.out_linear(output)
        output = F.leaky_relu(output, negative_slope=0.2)
        output = F.relu(output)
        return output

class DREMModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, num_layers, relation_cnt, dropout):
        super(DREMModel, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(DREMLayer(in_features, hidden_features, num_heads,relation_cnt, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(DREMLayer(hidden_features, hidden_features, num_heads,relation_cnt, dropout))
        self.layers.append(DREMLayer(hidden_features, out_features, num_heads, relation_cnt,dropout))

    def forward(self, x, adj):
        hidden_list=[]
        adj=adj.to_dense()
        for layer in self.layers:
            x = layer(x, adj)
            hidden_list.append(x)
        return hidden_list
