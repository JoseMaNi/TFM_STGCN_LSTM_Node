import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, to_torch_csr_tensor
from torch_geometric.utils.repeat import repeat


class GraphUNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            depth: int,
            pool_ratios: Union[float, List[float]] = 0.5,
            sum_res: bool = True,
            act: Union[str, Callable] = 'relu',
            sequence_length: int = 5,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res
        self.sequence_length = sequence_length

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved = True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved = True))

        self.lstm_blocks = torch.nn.ModuleList()
        for _ in range(depth):
            self.lstm_blocks.append(torch.nn.LSTM(channels, channels, batch_first = True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved = True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved = True))

        self.final_lstm = torch.nn.LSTM(channels, channels, batch_first = True)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()
        for lstm in self.lstm_blocks:
            for layer in lstm._all_weights:
                for param in layer:
                    if 'weight' in param:
                        torch.nn.init.xavier_uniform_(lstm.__getattr__(param))
        for layer in self.final_lstm._all_weights:
            for param in layer:
                if 'weight' in param:
                    torch.nn.init.xavier_uniform_(self.final_lstm.__getattr__(param))

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: OptTensor = None, mask: OptTensor = None) -> Tensor:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            if mask is not None:
                sequences = self.create_sequences(res, mask)
                if sequences.size(0) > 0:
                    res_lstm, _ = self.lstm_blocks[j](sequences)
                    res_lstm = res_lstm[:, -1, :]  # Último estado oculto
                    res_updated = res.clone()
                    res_updated[mask] = res_lstm
                else:
                    res_updated = res
            else:
                res_updated = res

            up = torch.zeros_like(res)
            up[perm] = x

            x = res_updated + up if self.sum_res else torch.cat((res_updated, up), dim = -1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        if mask is not None:
            sequences = self.create_sequences(x, mask)
            if sequences.size(0) > 0:
                x_lstm, _ = self.final_lstm(sequences)
                x_lstm = x_lstm[:, -1, :]
                x_final = x.clone()
                x_final[mask] = x_lstm
            else:
                x_final = x
        else:
            x_final = x

        return x_final

    def create_sequences(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Crea secuencias temporales para los nodos marcados.

        Args:
            x (Tensor): Tensor de entrada de forma [num_nodes, num_features]
            mask (Tensor): Tensor booleano que marca los nodos a procesar

        Returns:
            Tensor: Secuencias temporales de forma [num_masked_nodes, sequence_length, num_features]
        """
        masked_indices = torch.where(mask)[0]
        sequences = []

        for idx in masked_indices:
            start_idx = max(0, idx - self.sequence_length + 1)
            sequence = x[start_idx:idx + 1]

            # Rellenar con ceros si es necesario
            if sequence.size(0) < self.sequence_length:
                padding = torch.zeros(self.sequence_length - sequence.size(0), x.size(1), device = x.device)
                sequence = torch.cat([padding, sequence])

            sequences.append(sequence)

        if sequences:
            return torch.stack(sequences)
        else:
            return torch.empty(0, self.sequence_length, x.size(1), device = x.device)

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes = num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size = (num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')