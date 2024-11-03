
try:
    from MODEL.preprocessing_G import *
    from MODEL.model_funcs import *
    from MODEL.GNN_LSTM_MODEL import *
    from MODEL.Optimitation_and_plotting import proceso_optuna
    # from Optimitation_and_plotting import summary

except Exception as e:
    raise e

from CONSTANTS import *

import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ((T, N, F),
     (X_train,y_train, mask_train, edge_index) ,
     (train_loader, val_loader,test_loader))= prepare_dataset(device)

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

    print('Inicialización del modelo')
    print('EPOCHS DE ENTRENO: ', num_epochs)
    print('CARACTERISTICAS DE SALIDA ESPERADAS :')
    print('output_0: Tiempo accidente ' ,len(num_acc),
          ' | output_1: Personas accidentadas ', lim_accidentados + 1,
          ' | output_2: Gravedad accidente ',len(gravedad))

    model = STGCN_LSTM_Node(
        in_channels = F,
        hidden_channels = hidden,
        out_channels = out,  # Número de categorías para cada salida
        num_nodes = N,
        seq_length = timewindow,
        dropouts = dropouts
    ).to(device)

    num_outputs = len(model.out_channels)
    criterions = [nn.CrossEntropyLoss() for _ in range(num_outputs)]

    all_metrics_df, time_df = train_model(model = model,
                            train_loader = train_loader, val_loader = val_loader, criterions = criterions,
                            scheduler_factor=sch_factor,lr=lr,weight_decay = weight_decay,num_epochs = num_epochs,
                            loss_weights = loss_weights_,device = device)

    torch.cuda.empty_cache()

    torch.save(model.state_dict(), MODEL_FILE)
    print("Métricas guardadas en CSV y Pickle.")
    model.to(device)

    model.eval()
    print('Realizar predicciones en el conjunto de test')

    evaluate_model(model = model,val_loader =test_loader, criterions = criterions,
                   loss_weights = loss_weights_, device = device,testing_flag = True)

    all_metrics_df.to_csv(METRICS_FILE, index = False)
    time_df.to_csv(TIME_FILE, index = False)
