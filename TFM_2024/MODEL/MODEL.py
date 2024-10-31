
import gc

import torch.cuda

try:
    from preprocessing_G import *
    from model_funcs import *
except Exception as e:
    from MODEL.preprocessing_G import *
    from MODEL.model_funcs import *

class STGCN_LSTM_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, seq_length,dropouts=[0.25,0.25]):
        print('STGCN_LSTM_Node')
        super(STGCN_LSTM_Node, self).__init__()
        # print(in_channels, hidden_channels, out_channels, num_nodes, seq_length)
        dropout0, dropout_lstm = dropouts

        self.in_channels=in_channels
        self.hidden_channels=hidden_channels
        self.out_channels= out_channels
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.dropout1= nn.Dropout(dropout0)
        # self.dropout2= nn.Dropout(dropout0)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.fc_per_node = nn.ModuleList([
                nn.Linear(hidden_channels, hidden_channels) for _ in range(num_nodes)
            ])

        self.lstm_per_node = nn.ModuleList(
            [nn.LSTM(hidden_channels, hidden_channels, batch_first=True) for _ in range(num_nodes)])

        self.dropout_lstm_per_node = nn.ModuleList(
            [nn.Dropout(dropout_lstm) for _ in range(num_nodes)])

        self.fc_per_node_out = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_channels, out_channels[i]) for i in range(len(out_channels))
            ]) for _ in range(num_nodes)
        ])
        # print(self.fc_per_node[0])

    def forward(self, data):

        x, edge_index, masks,y = data.x, data.edge_index, data.mask,data.y
        del data
        gc.collect()
        torch.cuda.empty_cache()
        # Asumimos que x tiene forma [T, N, F], edge_index [2, E], y masks [T, N]

        max_events = int(masks.sum(dim=1).max().item())

        # edge_index0 = edge_index[0]
        #
        # x_flat = x.view(-1, x.size(2))  # [T * N, F] => Trata todos los nodos de todos los tiempos como una única lista
        # h = F_.relu(self.conv1(x_flat, edge_index0))  # [T * N, hidden_channels]
        # h = F_.relu(self.conv2(h, edge_index0))
        # x = h.view(x.size(0), x.size(1), -1)

        h_list = []
        for t in range(x.size(0)):  # Loop over time steps
            x_t = x[t]  # Shape: [N, F] for this time step
            h_t = F_.relu(self.conv1(x_t, edge_index[t]))  # Conv at time step t
            h_t = F_.relu(self.conv2(h_t, edge_index[t]))  # Another conv
            h_list.append(h_t)
        print('~~~~')
        x = torch.stack(h_list, dim = 0)
        del h_t , h_list
        gc.collect()
        torch.cuda.empty_cache()
        x = self.dropout1(x)
        print('~~~~')


        outputs = []
        filtered_targets = []

        for i in range(x.size(1)):  # Para cada nodo
            # Generar secuencias para el nodo actual
            node_features = x[:, i, :]  # [T, hidden_channels]
            node_mask = masks[:, i]  # [T]
            node_sequences = generate_sequences_node(node_features, node_mask, self.seq_length)

            # Aplicar LSTM
            lstm_out, _ = self.lstm_per_node[i](node_sequences)
            lstm_out = F_.relu(lstm_out)

            lstm_out_do = self.dropout_lstm_per_node[i](lstm_out)
            lstm_out_fc = self.fc_per_node[i](lstm_out_do)

            # Aplicar capas fully connected
            node_output = [fc_(lstm_out_fc[:, -1, :]) for fc_ in self.fc_per_node_out[i]]
            node_output_padded = [self._pad_or_truncate(out, max_events) for out in node_output]

            # Filtrar salidas por la máscara
            outputs.append(node_output_padded)

            # Filtrar las etiquetas para que coincidan con los pasos temporales relevantes
            node_targets = y[:, i, :]  # Etiquetas del nodo i
            filtered_target = node_targets[node_mask]  # Filtramos usando la máscara

            # Paddear las etiquetas para que tengan la misma longitud
            filtered_target_padded = self._pad_or_truncate(filtered_target, max_events)
            filtered_targets.append(filtered_target_padded)

        # Convertir listas en tensores y devolver ambas salidas filtradas y etiquetas filtradas
        outputs_stacked = [torch.stack([out[i] for out in outputs], dim = 1) for i in range(len(self.fc_per_node_out[0]))]
        filtered_targets_stacked = torch.stack(filtered_targets,dim = 1)

        return outputs_stacked, filtered_targets_stacked

    def _pad_or_truncate(self, tensor, target_length):
        if tensor.size(0) < target_length:
            padding = torch.zeros(target_length - tensor.size(0), tensor.size(1), device = tensor.device)
            return torch.cat([tensor, padding], dim = 0)
        else:
            return tensor[:target_length]
