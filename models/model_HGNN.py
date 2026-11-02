import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from models.layers.cross_attention import FeedForward, MMAttentionLayer
from models.layers.fusion import GraphFusion, AlignFusion
from models.layers.layers import *
from models.layers.sheaf_builder import *
from torch_scatter import scatter_mean
from .util import initialize_weights
from .util import NystromAttention
from .util import SNN_Block
from .util import MultiheadAttention
from collections import defaultdict
from models.layers.relation_gat import RelationAwareGAT
from torch_geometric.nn import RGCNConv


class EnhancedRGCN(nn.Module):
    def __init__(self, in_channels=256, out_channels=256,
                 num_relations=2, num_bases=None,
                 num_layers=2, dropout=0.25):
        super().__init__()
        self.num_layers = num_layers

        self.conv1 = RGCNConv(in_channels, out_channels, num_relations, num_bases)
        self.norm1 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)


        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.res_proj = None


    def forward(self, x, edge_index, edge_type):
        h1_in = self.norm1(x)
        h1 = F.relu(self.conv1(h1_in, edge_index, edge_type))
        h1 = self.dropout1(h1)

        if self.res_proj is not None:
            h1 = h1 + self.res_proj(x)
        else:
            h1 = h1 + x


        return h1



class CDGPath(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 fusion="concat", model_size="small", graph_type="RGCN"):
        super(CDGPath, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.graph_type = graph_type

        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }

        # ============ Pathomics Embedding ============
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)

        # ============ Graph ============
        if self.graph_type == "RGCN":
            self.patch_rgnn = EnhancedRGCN(
                in_channels=256,
                out_channels=256,
                num_relations=2,
                num_bases=None,
                dropout=0.5
            )


        # ============ Genomics Embedding ============
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # ============ Pre-fusion: Mono-Holo Confidence ============
        g_dim = self.size_dict["genomics"][model_size][-1]
        p_dim = self.size_dict["pathomics"][model_size][-1]
        g_num = 6
        p_num = 4096

        self.ConfidNet_g = nn.Sequential(
            nn.Linear(g_num * g_dim, g_num * g_dim * 2),
            nn.ReLU(),
            nn.Linear(g_num * g_dim * 2, g_num * g_dim),
            nn.ReLU(),
            nn.Linear(g_num * g_dim, 1),
            nn.Sigmoid()
        )
        self.ConfiNet_p_pre = nn.Linear(p_dim, 1)
        self.ConfidNet_p = nn.Sequential(
            nn.Linear(p_num, p_num * 2),
            nn.ReLU(),
            nn.Linear(p_num * 2, p_num),
            nn.ReLU(),
            nn.Linear(p_num, 1),
            nn.Sigmoid()
        )

        # ============ AlignFusion ============
        self.attention_fusion = AlignFusion(
            embedding_dim=g_dim,
            num_heads=4,
            num_pathways=g_num
        )

        self.feed_forward = FeedForward(g_dim, dropout=0.25)
        self.layer_norm = nn.LayerNorm(g_dim)

        hidden = self.size_dict["genomics"][model_size]
        self.mm = nn.Sequential(
                nn.Linear(hidden[-1] * 2, hidden[-1] // 2),
                nn.ReLU()
            )
        self.classifier = nn.Linear(hidden[-1] // 2, self.n_classes)

        self.apply(initialize_weights)

    def compute_mono_holo_weights(self, path_feat, gene_feat):
        """计算 Mono-Holo Confidence 权重"""
        B = 1

        if path_feat.shape[1] == 4096:
            tmp = self.ConfiNet_p_pre(path_feat).view(B, -1)
            path_tcp = self.ConfidNet_p(tmp)
        else:
            tmp = self.ConfiNet_p_pre(path_feat).view(B, -1)
            path_tcp = torch.sigmoid(tmp.mean(dim=1, keepdim=True))

        gene_tcp = self.ConfidNet_g(gene_feat.view(B, -1))

        eps = 1e-8
        path_holo = torch.log(gene_tcp + eps) / (torch.log(path_tcp * gene_tcp + eps) + eps)
        gene_holo = torch.log(path_tcp + eps) / (torch.log(path_tcp * gene_tcp + eps) + eps)

        cb_path = path_tcp.detach() + path_holo.detach()
        cb_gene = gene_tcp.detach() + gene_holo.detach()


        w_all = torch.stack((cb_path, cb_gene), dim=1).squeeze(-1)
        w_all = F.softmax(w_all, dim=1)

        w_path = w_all[:, 0:1]
        w_gene = w_all[:, 1:2]

        return w_path, w_gene

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]

        # ============ 1. 特征提取 ============
        genomics_features = [self.genomics_fc[idx](sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)

        pathomics_features = self.pathomics_fc(x_path)[0]

        # ============ 2. Pre-fusion: Mono-Holo ============
        wsi_f = pathomics_features.unsqueeze(0)
        #wsi_f = pathomics_features
        w_path, w_gene = self.compute_mono_holo_weights(wsi_f, genomics_features)

        # ============ 3. Graph Learning ============
        graph = kwargs["graph"]
        device = pathomics_features.device

        edge_index_space = graph.edge_index.to(device)
        edge_index_feat = graph.edge_latent.to(device)
        edge_index = torch.cat([edge_index_space, edge_index_feat], dim=1)

        edge_type = torch.cat([
            torch.zeros(edge_index_space.size(1), dtype=torch.long),
            torch.ones(edge_index_feat.size(1), dtype=torch.long)
        ], dim=0).to(device)

        pathomics_features = self.patch_rgnn(
            pathomics_features,
            edge_index=edge_index,
            edge_type=edge_type
        ).unsqueeze(0)

        # # ============ 4. 应用 Pre-fusion 权重 ============
        # pathomics_weighted = pathomics_proto * w_path.unsqueeze(-1)

        pathomics_weighted = pathomics_features * w_path.unsqueeze(-1)
        genomics_weighted = genomics_features * w_gene.unsqueeze(-1)

        # ============ 5. AlignFusion ============
        token = torch.cat((genomics_weighted, pathomics_weighted), dim=1)
        token_cross = self.attention_fusion(token=token)

        token_cross = self.feed_forward(token_cross)
        token_cross = self.layer_norm(token_cross)

        gene_embed = token_cross[:, :6, :]
        wsi_embed = token_cross[:, 6:, :]

        gene_pooled = gene_embed.mean(dim=1)
        wsi_pooled = wsi_embed.mean(dim=1)
        fusion = self.mm(torch.cat([gene_pooled, wsi_pooled], dim=1))


        logits = self.classifier(fusion)

        return logits