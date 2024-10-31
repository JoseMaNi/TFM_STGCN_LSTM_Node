from sklearn.decomposition import PCA
from funcs import *
import torch

def preprocess_graph(G, LABEL_COLS, FEATURE_COLS):
    T = len(G.nodes[0]['medidas'])  # Número de instantes de tiempo
    N = len(G.nodes)  # Número de nodos
    Y = len(LABEL_COLS)  # Número de características por nodo

    Ys = []
    masks = []
    indexes = []

    for node ,data_ in G.nodes.items():

        # Guardamos las fechas antes de eliminarlas
        fechas = data_['medidas']['fecha']
        # Guardamos los índices (fechas) para cada nodo
        indexes.append(fechas)
        # data_= data_['medidas'].drop(DROP_COLS,axis=1)
        # Extraemos las etiquetas y las máscaras

        Y_vals=data_['medidas'][LABEL_COLS].values
        mask_vals = data_['medidas']['pad'].values

        Ys.append(Y_vals)
        masks.append(mask_vals)

        # Actualizamos el nodo con solo las características relevantes
        G.nodes[node]['medidas'] = data_['medidas'][FEATURE_COLS]

    Ys_t = torch.zeros(N, T, Y)
    masks_t = torch.zeros(N, T, dtype=torch.bool)

    for i, (y, mask) in enumerate(zip(Ys, masks)):
        Ys_t[i, :, :] = torch.tensor(y, dtype=torch.float32)
        masks_t[i, :] = torch.tensor(mask, dtype=torch.bool)

    return G, Ys_t, masks_t, indexes

def graph_to_tensor(G_,Ys, masks):

    T = len(G_.nodes[0]['medidas'])  # Número de instantes de tiempo
    N = len(G_.nodes)  # Número de nodos
    F = len(G_.nodes[0]['medidas'].columns)  # Número de características por nodo
    print(' TxNxF ')
    print(T,N,F)
    features = torch.zeros(T, N, F)

    for  node, data_ in G_.nodes.items():
        features_np = np.array(G_.nodes[node]['medidas'].values, dtype = np.float32)
        features[:, node, :] = torch.tensor(features_np, dtype = torch.float32)

    Ys = Ys.transpose(0, 1)
    masks = masks.transpose(0, 1)

    edge_index = torch.tensor(list(G_.edges())).t().contiguous()


    return features,Ys,masks, edge_index

def preprocess_graph_to_tensors_0 (G, LABEL_COLS, FEATURE_COLS):

    T = len(G.nodes[0]['medidas'])  # Número de instantes de tiempo
    N = len(G.nodes)  # Número de nodos
    Y = len(LABEL_COLS)  # Número de etiquetas por nodo

    features_np = np.zeros((T, N, len(FEATURE_COLS)), dtype = np.float32)
    Ys_np = np.zeros((T, N, Y), dtype = np.float32)
    masks_np = np.zeros((T, N), dtype = bool)

    indexes = []
    for i, (node, data_) in enumerate(G.nodes.items()):
        medidas = data_['medidas']
        fechas = medidas['fecha']
        indexes.append(fechas)

        Y_vals = medidas[LABEL_COLS].values
        mask_vals = medidas['pad'].values

        node_features = medidas[FEATURE_COLS].values

        features_np[:, i, :] = node_features
        Ys_np[:, i, :] = Y_vals
        masks_np[:, i] = mask_vals

    features_t = torch.tensor(features_np)
    Ys_t = torch.tensor(Ys_np)
    masks_t = torch.tensor(masks_np)

    edge_index = torch.tensor(list(G.edges())).t().contiguous()

    return features_t, Ys_t, masks_t, edge_index, indexes


def preprocess_graph_to_tensors(G, LABEL_COLS, FEATURE_COLS, pca=False):
    T = len(G.nodes[0]['medidas'])  # Número de instantes de tiempo
    N = len(G.nodes)  # Número de nodos
    Y = len(LABEL_COLS)  # Número de etiquetas por nodo

    # Crear matrices numpy para las características, etiquetas y máscara
    features_np = np.zeros((T, N, len(FEATURE_COLS)), dtype=np.float32)
    Ys_np = np.zeros((T, N, Y), dtype=np.float32)
    masks_np = np.zeros((T, N), dtype=bool)

    indexes = []
    for i, (node, data_) in enumerate(G.nodes.items()):
        medidas = data_['medidas']
        fechas = medidas['fecha']
        indexes.append(fechas)

        Y_vals = medidas[LABEL_COLS].values
        mask_vals = medidas['pad'].values

        node_features = medidas[FEATURE_COLS].values

        features_np[:, i, :] = node_features  # Características originales
        Ys_np[:, i, :] = Y_vals
        masks_np[:, i] = mask_vals

    # Si 'pca' es un número entero, aplicamos PCA para reducir las características
    if isinstance(pca, int) and pca > 0:
        print(f"Aplicando PCA para reducir las características a {pca} dimensiones.")
        # Aplicar PCA en el eje de características (reduce de F a pca dimensiones)
        features_flat = features_np.reshape(-1, features_np.shape[2])  # [T * N, F]
        pca_model = PCA(n_components=pca)
        features_reduced = pca_model.fit_transform(features_flat)  # [T * N, pca]
        features_np = features_reduced.reshape(T, N, pca)  # Reshape a [T, N, pca]

    # Convertir las matrices numpy a tensores de PyTorch
    features_t = torch.tensor(features_np)
    Ys_t = torch.tensor(Ys_np)
    masks_t = torch.tensor(masks_np)

    # Crear el tensor edge_index para las aristas del grafo
    edge_index = torch.tensor(list(G.edges())).t().contiguous()

    return features_t, Ys_t, masks_t, edge_index, indexes




def split_data_temporal(features, Ys, masks, indexes,days):
    # Asumimos que todos los nodos tienen las mismas fechas
    fechas = indexes[0]

    corte1 = fechas.iloc[0] + pd.Timedelta(days = days[0])
    corte2 = corte1 + pd.Timedelta(days = days[1])

    mascara_train = fechas < corte1
    mascara_val = (fechas >= corte1) & (fechas < corte2)
    mascara_test = fechas >= corte2

    X_train = features[mascara_train]
    X_val = features[mascara_val]
    X_test = features[mascara_test]

    y_train = Ys[mascara_train]
    y_val = Ys[mascara_val]
    y_test = Ys[mascara_test]

    mask_train = masks[mascara_train]
    mask_val = masks[mascara_val]
    mask_test = masks[mascara_test]

    return X_train, X_val, X_test, y_train, y_val, y_test, mask_train, mask_val, mask_test
