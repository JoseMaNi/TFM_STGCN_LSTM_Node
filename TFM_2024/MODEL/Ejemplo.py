

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
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if __name__ == "__main__":

    G=main()

    FEATURE_COLS = G.nodes[0]['medidas'].columns.drop(DROP_COLS+LABEL_COLS)
    T0 = G.nodes[0]['medidas'].shape[0]
    F0 = G.nodes[0]['medidas'].shape[1]
    N0 = len(G.nodes())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(f' {N0} Nodos , {F0} Caracteristicas, {T0} Pasos')
    features,Ys, masks, edge_index, indexes = preprocess_graph_to_tensors (G, LABEL_COLS, FEATURE_COLS,pca=finalPCAdim)
    del G
    # [print(*zip(unique_values, counts)) for unique_values, counts in (torch.unique(Ys[:, n, :], return_counts=True) for n in range(N0))]
    print(features.shape)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     mask_train, mask_val, mask_test) = split_data_temporal(features, Ys, masks, indexes,periodos)

    del features, Ys, masks
    gc.collect()
    # Configuración del dispositivo
    print(f"Using device: {device}")
    # Crear batches
    train_data = create_batch(X_train, edge_index, mask_train, y_train, device)
    val_data = create_batch(X_val, edge_index, mask_val, y_val, device)
    test_data = create_batch(X_test, edge_index, mask_test, y_test, device)
    print('------------')
    # Creación de los DataLoaders
    train_loader = DataLoader([train_data], batch_size = batch_size, shuffle = False)#,pin_memory=True)
    val_loader = DataLoader([val_data], batch_size = batch_size, shuffle = False)#,pin_memory=True)
    test_loader = DataLoader([test_data], batch_size = batch_size, shuffle = False)#,pin_memory=True)

    del X_val, X_test, y_val, y_test, mask_val, mask_test
    del train_data,val_data,test_data
    torch.cuda.empty_cache()
    gc.collect()

    T, N, F = X_train.shape


    # print(train_loader)
    # 1: {'learning_rate': 0.0013477769929348268, 'hidden_channels': 128, 'dropout0': 0.420153227000495,
    #     'dropout_lstm': 0.36839816000093045}
    if OPTUNA_FLAG:
        if OPTUNA_FILE not in os.listdir():

            best_lr, best_weight_decay,best_hidden, best_dropout0,best_dropout_lstm=proceso_optuna(X_train = X_train,
                                                              y_train = y_train,
                                                              mask_train = mask_train,
                                                              val_loader = val_loader,
                                                              edge_index = edge_index,device = device, out=out, shapeNF = [N,F],
                                                              timewindow = timewindow, subset_prop=prop_optuna,
                                                              num_epochs = num_epochs_optuna, n_trials=n_trials,phases = phases_optuna)
            best_dropout = [best_dropout0, best_dropout_lstm]
            lr=best_lr
            weight_decay = best_weight_decay
            hidden = best_hidden
            dropouts = best_dropout

            with open(OPTUNA_FILE, 'wb') as f:

                pickle.dump([best_lr, best_hidden, best_dropout0, best_dropout_lstm], f)
            print('SE GUARDA OPTUNA ', os.getcwd())
        else:
            print('HAY HIPERPARAMETROS, SE CARGA')

            with open(OPTUNA_FILE, 'rb') as f:
                best_lr, best_hidden, best_dropout0, best_dropout_lstm = pickle.load(f)
    del X_train, y_train, mask_train
    gc.collect()
    torch.cuda.empty_cache()


    # Inicialización del modelo

    print('EPOCHS DE ENTRENO: ', num_epochs)
    print("CARACTERISTICAS DE SALIDA ESPERADAS :", [len(num_acc), lim_accidentados + 1, len(gravedad)])






    model = STGCN_LSTM_Node(
        in_channels = F,
        hidden_channels = hidden,
        out_channels = out,  # Número de categorías para cada salida
        num_nodes = N,
        seq_length = timewindow,
        dropouts = dropouts
    ).to(device)

    # summary(model,train_loader,F)

    num_outputs = len(model.out_channels)

    criterions = [nn.CrossEntropyLoss() for _ in range(num_outputs)]


    all_metrics_df, time_df = train_model(model = model,
                            train_loader = train_loader, val_loader = val_loader, criterions = criterions,
                            lr=lr,weight_decay = weight_decay, num_epochs = num_epochs, loss_weights = loss_weights_,
                            device = device)

    torch.cuda.empty_cache()
    (test_loss,     test_acc,
     test_metrics) = evaluate_model(model = model,
                                  val_loader = test_loader,
                                  criterions = criterions,
                                  loss_weights = loss_weights_,
                                  device = device)

    torch.save(model.state_dict(), MODEL_FILE)


    all_metrics_df.to_csv(METRICS_FILE, index = False)
    time_df.to_csv(TIME_FILE, index = False)
    # classification_report_df.to_csv(REPORTS_FILE,index = False)



    print("Métricas guardadas en CSV y Pickle.")
    model.to(device)

    # Realizar predicciones en el conjunto de test
    model.eval()
