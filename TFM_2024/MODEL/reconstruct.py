import time




try:
    from preprocessing_G import *
    from model_funcs import *
    from MODEL import *
    from Optimitation_and_plotting import proceso_optuna
    from Optimitation_and_plotting import summary

except Exception as e:
    from MODEL.preprocessing_G import *
    from MODEL.model_funcs import *
    from MODEL.MODEL import *
    from MODEL.Optimitation_and_plotting import summary
    from MODEL.Optimitation_and_plotting import proceso_optuna

# from funcs import *
from main import main
from CONSTANTS import *
# from MODEL import *

def measure_transfer_time(data, device):
    start = time.time()
    data = data.to(device)
    torch.cuda.synchronize()  # Asegura que la operación GPU ha terminado
    gpu_time = time.time() - start

    start = time.time()
    data = data.to('cpu')
    cpu_time = time.time() - start

    return gpu_time, cpu_time





if __name__ == "__main__":

    G=main()

    FEATURE_COLS = G.nodes[0]['medidas'].columns.drop(DROP_COLS+LABEL_COLS)
    T0 = G.nodes[0]['medidas'].shape[0]
    F0 = G.nodes[0]['medidas'].shape[1]
    N0 = len(G.nodes())


    print(f' {N0} Nodos , {F0} Caracteristicas, {T0} Pasos')
    features,Ys, masks, edge_index, indexes = preprocess_graph_to_tensors (G, LABEL_COLS, FEATURE_COLS)
    del G
    # [print(*zip(unique_values, counts)) for unique_values, counts in (torch.unique(Ys[:, n, :], return_counts=True) for n in range(N0))]
    print(features.shape)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     mask_train, mask_val, mask_test) = split_data_temporal(features, Ys, masks, indexes,periodos)

    del features, Ys, masks
    gc.collect()
    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Crear batches
    train_data = create_batch(X_train, edge_index, mask_train, y_train, device)
    val_data = create_batch(X_val, edge_index, mask_val, y_val, device)
    test_data = create_batch(X_test, edge_index, mask_test, y_test, device)
    print('------------')

    # Creación de los DataLoaders
    train_loader = DataLoader([train_data], batch_size = 1, shuffle = False)
    val_loader = DataLoader([val_data], batch_size = 1, shuffle = False)
    test_loader = DataLoader([test_data], batch_size = 1, shuffle = False)






# Ejemplo de uso
sample_batch = next(iter(train_loader))
to_gpu_time, to_cpu_time = measure_transfer_time(sample_batch, device)

print(f"Tiempo para mover a GPU: {to_gpu_time:.4f} segundos")
print(f"Tiempo para mover a CPU: {to_cpu_time:.4f} segundos")