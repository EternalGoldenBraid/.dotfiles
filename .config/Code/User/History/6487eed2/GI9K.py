# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

from src.slaps.graph_generator import MLP_generator as MLP_graph_gen
from src.slaps.graph_generator import FullParam, MLP_Diag
from src.slaps.layers import GCNConv_dense, GCNConv_dgl

from src.slaps.utils import symmetrize, normalize

from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import GATv2Conv, GATConv


class GCN_DAE(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k, knn_metric, i_,
                 non_linearity, normalization, mlp_h, mlp_epochs, gen_mode, sparse, mlp_act):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dgl(hidden_dim, nclasses))

        else:
            self.layers.append(GCNConv_dense(in_dim, hidden_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dense(hidden_dim, nclasses))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.k = k
        self.knn_metric = knn_metric
        self.i = i_
        self.non_linearity = non_linearity
        self.normalization = normalization
        self.nnodes = features.shape[0]
        self.mlp_h = mlp_h
        self.mlp_epochs = mlp_epochs
        self.sparse = sparse

        if gen_mode == 0:
            self.graph_gen = FullParam(features.detach().cpu(), non_linearity, k, knn_metric, self.i, sparse).cuda()
        elif gen_mode == 1:
            self.graph_gen = MLP_graph_gen(2, features.shape[1], math.floor(math.sqrt(features.shape[1] * self.mlp_h)),
                                 self.mlp_h, mlp_epochs, k, knn_metric, self.non_linearity, self.i, self.sparse,
                                 mlp_act, features=features.detach().cpu()).cuda()

        elif gen_mode == 2:
            self.graph_gen = MLP_Diag(nlayers=2,  k=k, isize=features.shape[1],
            knn_metric=knn_metric, non_linearity=self.non_linearity,
            i=self.i, sparse=sparse, mlp_act=mlp_act).cuda()

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        if not self.sparse:
            Adj_ = symmetrize(Adj_)
            Adj_ = normalize(Adj_, self.normalization, self.sparse)
        return Adj_

    def forward(self, features, x):  # x corresponds to masked_fearures
        Adj_ = self.get_adj(features)
        if self.sparse:
            Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x, Adj_


class GCN(nn.Module):
    """ Fixed adjacency (it seems)"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.layers.append(GCNConv(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse

    def forward(self, x):

        if self.sparse:
            Adj = self.Adj
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x


class GCN_C(nn.Module):
    """
    The difference between GCN and GCN_C is that GCN_C uses the same adjacency matrix for all layers
    which is apparent in the forward function between rows 40 and 50 in GCN_C and rows 40 and 45 in GCN
    
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GCN_C, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

    def forward(self, x, adj_t):
        if self.sparse:
            Adj = adj_t
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(adj_t)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x

class GAT_C(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GCN_C, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

    def forward(self, x, adj_t):
        if self.sparse:
            Adj = adj_t
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(adj_t)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x
    
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads, concat, device):
        super(GAT, self).__init__()

        self.layers = nn.ModuleList()

        # self.layers.append(GATv2Conv(in_channels, hidden_channels, 
        self.layers.append(GATConv(in_channels, hidden_channels, 
                           heads=heads, dropout=dropout, concat=concat))
        for i in range(num_layers - 2):
            # self.layers.append(GATv2Conv(hidden_channels, hidden_channels, 
            self.layers.append(GATConv(hidden_channels, hidden_channels, 
                           heads=heads, dropout=dropout, concat=concat))
        self.layers.append(GATConv(hidden_channels, out_channels))
        # self.layers.append(GATv2Conv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.device = device
        
    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.layers[:-1]):
            x = F.relu(conv(x, edge_index=edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x, (edge_index_, att_weights) = self.layers[-1](x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.layers[-1](x, edge_index, edge_attr=edge_attr)

        # return x, edge_index_, att_weights
        return x
                                           
from transformers import BertTokenizer, BertForMaskedLM
import torch
class BertMLMEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model.language_encoders.bert.model_name)
        self.model = BertForMaskedLM.from_pretrained(config.model.language_encoders.bert.model_name)

        for n, p in self.model.named_parameters():
            p.requires_grad = False

    def mask_tokens(self, inputs, mask_prob):
        """
        Create a mask for a certain percentage of tokens in the input sequence.
        """
        mask = torch.bernoulli(torch.full(inputs.shape, mask_prob)).bool()
        labels = torch.where(mask, inputs, -100)
        masked_inputs = inputs.clone()
        masked_inputs[mask] = self.tokenizer.mask_token_id
        return masked_inputs, labels

    def __call__(self, data, mask_prob=0.15):
        input_ids = self.tokenizer.encode(data, return_tensors='pt',
                            max_length=self.config.model.language_encoders.bert.max_seq_length,
                            truncation=True, padding='max_length')

        # Create mask and replace tokens with [MASK]
        masked_input_ids, labels = self.mask_tokens(input_ids, mask_prob)

        # Predict the masked tokens
        with torch.no_grad():
            outputs = self.model(masked_input_ids, labels=labels)
            logits = outputs.logits

        return logits