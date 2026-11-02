import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class RelationAwareGAT(MessagePassing):
    def __init__(self, in_dim, out_dim, num_relations=2, heads=4, dropout=0.5):
        super().__init__(aggr='add',node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.num_relations = num_relations

        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)

        # 每种 relation 一个 attention 向量
        self.att = nn.Parameter(torch.Tensor(num_relations, heads, 2 * out_dim))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_type):
        """
        x: [N, C]
        edge_index: [2, E]
        edge_type: [E]  (0: space, 1: feature)
        """
        H, C = self.heads, self.out_dim
        x = self.lin(x).view(-1, H, C)

        return self.propagate(edge_index, x=x, edge_type=edge_type)

    def message(self, x_i, x_j, edge_index_i, edge_type):
        """
        x_i, x_j: [E, H, C]
        """
        # 根据 relation 选 attention 参数
        att = self.att[edge_type]   # [E, H, 2C]
        alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=1)  # multi-head average
