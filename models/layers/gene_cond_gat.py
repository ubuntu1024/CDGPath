import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class GeneConditionedGAT(MessagePassing):
    def __init__(self, in_dim, gene_dim, out_dim, heads=4, dropout=0.5):
        super().__init__(aggr='add',node_dim=0)
        self.heads = heads
        self.out_dim = out_dim

        self.lin_x = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_g = nn.Linear(gene_dim, heads, bias=False)

        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_dim))
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_x.weight)
        nn.init.xavier_uniform_(self.lin_g.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, gene_embed):
        """
        x: [N, C]
        gene_embed: [G, C]  (e.g. 6 pathways)
        """
        H, C = self.heads, self.out_dim
        x = self.lin_x(x).view(-1, H, C)

        # gene condition vector
        g = gene_embed.mean(dim=0, keepdim=True)  # [1, C]
        g = self.lin_g(g).view(1, H)               # [1, H]

        return self.propagate(edge_index, x=x, gene_gate=g)

    def message(self, x_i, x_j, edge_index_i, gene_gate):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = alpha * gene_gate
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=1)
