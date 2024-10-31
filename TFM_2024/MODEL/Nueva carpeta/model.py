

import torch.nn as nn
import torch_geometric.nn as gnn
# from typing import Callable, List, Union

import torch

from torch_geometric.nn.resolver import activation_resolver
# from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat

class GraphUNet_LSTM(nn.Module):


    def __init__(self, in_channels, hidden_channels, out_channels, depth, pool_ratios = 0.5, sum_res = True,
                 act = 'relu'):
        super(GraphUNet_LSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        # Capas de convolución de grafos (GCN) y LSTM
        self.down_convs = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()  # LSTM tras cada GCN

        # Capas de pooling
        self.pools = nn.ModuleList()

        # Primera capa GCN (antes del bucle)
        self.down_convs.append(gnn.GCNConv(in_channels, hidden_channels, improved = True))

        # Primera capa LSTM (antes del bucle)
        self.lstm_layers.append(nn.LSTM(hidden_channels, hidden_channels, batch_first = True))

        # Luego, en cada iteración del bucle: GCN + LSTM + Pool
        for i in range(depth):
            # TopK pooling para cada nivel
            self.pools.append(gnn.TopKPooling(hidden_channels, self.pool_ratios[i]))
            # Convolución gráfica para cada nivel
            self.down_convs.append(gnn.GCNConv(hidden_channels, hidden_channels, improved = True))
            # LSTM para cada nivel
            self.lstm_layers.append(nn.LSTM(hidden_channels, hidden_channels, batch_first = True))

        # Capas de upsampling (decodificación)
        in_channels = hidden_channels if sum_res else 2 * hidden_channels
        self.up_convs = nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(gnn.GCNConv(in_channels, hidden_channels, improved = True))
        self.up_convs.append(gnn.GCNConv(in_channels, out_channels, improved = True))

        # Inicializar parámetros
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for lstm in self.lstm_layers:
            for layer in lstm.parameters():
                if layer.dim() > 1:
                    nn.init.xavier_uniform_(layer)  # Inicializar parámetros de LSTM
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch = None, pad_mask = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        edge_weight = x.new_ones(edge_index.size(1))

        # Lista para almacenar los resultados intermedios (GCN-LSTM) para skip connections
        xs = []

        # Primera convolución gráfica (fuera del bucle)
        x = self.down_convs[0](x[:, -1], edge_index, edge_weight)
        x = self.act(x)

        # Primera LSTM (fuera del bucle)
        x = x.unsqueeze(0)
        if pad_mask is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(x, pad_mask, batch_first = True, enforce_sorted = False)
            x, _ = self.lstm_layers[0](x_packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        else:
            x, _ = self.lstm_layers[0](x)
        x = x.squeeze(0)

        # Guardamos el resultado para el primer skip connection
        xs.append(x)

        # Bucle de GCN + LSTM + Pool
        edge_indices, edge_weights, perms = [edge_index], [edge_weight], []
        for i in range(1, self.depth + 1):
            # Convolución gráfica
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            # LSTM después de cada GCN
            x = x.unsqueeze(0)
            if pad_mask is not None:
                x_packed = nn.utils.rnn.pack_padded_sequence(x, pad_mask, batch_first = True, enforce_sorted = False)
                x, _ = self.lstm_layers[i](x_packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
            else:
                x, _ = self.lstm_layers[i](x)
            x = x.squeeze(0)

            # Pooling
            if i < self.depth:  # No hacemos pooling en el último nivel
                edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
                x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)

            # Guardar el resultado para skip connections
            if i < self.depth:
                xs.append(x)
                edge_indices.append(edge_index)
                edge_weights.append(edge_weight)
            perms.append(perm)

        # Upsampling (decodificación) con skip connections
        for i in range(self.depth):
            j = self.depth - 1 - i
            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x  # Proceso de unpooling

            # Conexión residual: suma o concatenación
            x = res + up if self.sum_res else torch.cat((res, up), dim = -1)

            # Convolución gráfica para la decodificación
            x = self.up_convs[i](x, edge_index, edge_weight)
            if i < self.depth - 1:
                x = self.act(x)

        return x

    @staticmethod
    def augment_adj(edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes = num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size = (num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight