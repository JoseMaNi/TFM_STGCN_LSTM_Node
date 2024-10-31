import pandas as pd
from sklearn.metrics import confusion_matrix
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary as summary_text
try:
    from preprocessing_G import *
    from model_funcs import *
    from MODEL import *
except Exception as e:
    from MODEL.preprocessing_G import *
    from MODEL.model_funcs import *
    from MODEL.MODEL import *

from CONSTANTS import *


def summary(model,data,numfeatures):
    writer = SummaryWriter()
    writer.add_graph(model, data)
    writer.close()

    summary_text(model, input_size = numfeatures)


def modeloptunaF1_wrapper(subset_train_loader, val_loader, out, shapeNF, timewindow, device, num_epochs,weights):
    def modeloptunaF1(trial):
        lr = trial.suggest_float('learning_rate', 0.0005, 0.01,log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.00001, 0.001,log=True)
        hidden = trial.suggest_categorical('hidden_channels', [64])
        dropout0 = trial.suggest_float('dropout0', 0.25, 0.45)
        dropout_lstm = trial.suggest_float('dropout_lstm', 0.25, 0.45)
        dropouts = [dropout0,dropout_lstm]
        N,F = shapeNF
        # Definir el modelo con los hiperparámetros sugeridos
        model = STGCN_LSTM_Node(
            in_channels = F,
            hidden_channels = hidden,
            out_channels = out,
            num_nodes = N,
            seq_length = timewindow,
            dropouts = dropouts  # Aplicar dropout sugerido por Optuna
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr = lr,weight_decay = weight_decay)
        criterions = [nn.CrossEntropyLoss() for _ in range(len(model.out_channels))]


        train_model(model = model, train_loader = subset_train_loader,val_loader = val_loader, criterions = criterions,
                    lr=lr,weight_decay = weight_decay, num_epochs = num_epochs,
                    loss_weights = weights, device= device,metrics=False)

        val_loss, avg_acc, _ = evaluate_model(model, val_loader, criterions,weights, device)
        torch.cuda.empty_cache()
        gc.collect()
        # return avg_acc
        return val_loss  # Optimizar la pérdida en validación

    return modeloptunaF1


def modeloptunaF2_wrapper(Opt_best, train_loader, val_loader, out, shapeNF, timewindow, device, num_epochs,weights):
    def modeloptunaF2(trial):
        lr = trial.suggest_float('learning_rate', Opt_best['learning_rate'] * 0.3, Opt_best['learning_rate'] * 3, log=True)
        weight_decay = trial.suggest_float('weight_decay', Opt_best['weight_decay'] * 0.3, Opt_best['weight_decay'] * 3, log=True)
        hidden = Opt_best['hidden_channels']
        dropout0 = float(trial.suggest_float('dropout0', Opt_best['dropout0'] * 0.8, Opt_best['dropout0'] * 1.15))
        dropout_lstm = float(trial.suggest_float('dropout_lstm', Opt_best['dropout_lstm'] * 0.8, Opt_best['dropout_lstm'] * 1.15))
        dropouts = [dropout0, dropout_lstm]
        # Definir el modelo con los hiperparámetros ajustados
        N,F = shapeNF

        model = STGCN_LSTM_Node(
            in_channels = F,
            hidden_channels = hidden,
            out_channels = out,
            num_nodes = N,
            seq_length = timewindow,
            dropouts = dropouts  # Dropout optimizado
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr = lr,weight_decay = weight_decay)
        criterions = [nn.CrossEntropyLoss() for _ in range(len(model.out_channels))]

        # Entrenamiento con más épocas y usando el conjunto completo de entrenamiento

        all_metrics_df, time_df= train_model(model = model, train_loader = train_loader,
                                                        val_loader = val_loader, criterions = criterions,
                                                        lr=lr,weight_decay = weight_decay, num_epochs = num_epochs,
                                                        loss_weights = weights, device= device,metrics=False)
        # Evaluación en validación
        val_loss, avg_acc, _= evaluate_model(model, val_loader, criterions, weights,device)
        torch.cuda.empty_cache()
        # return avg_acc
        return val_loss  # Minimizar la pérdida de validación

    return modeloptunaF2


def proceso_optuna(X_train, y_train, mask_train, val_loader, edge_index, device, out, shapeNF, timewindow,
                   num_epochs = [10,50], n_trials=[15,30],subset_prop=0.2,phases=True):
    print('OPTUNA')
    subset_size = int(subset_prop[0] * len(X_train))  # Subset del X%
    start_idx = torch.randint(low=0, high=int((0.99-subset_prop[0])*len(X_train)), size=(1,))

    # Crear el subset correlativo
    X_train_subset = X_train[start_idx:start_idx + subset_size]
    y_train_subset = y_train[start_idx:start_idx + subset_size]
    mask_train_subset = mask_train[start_idx:start_idx + subset_size]

    # Convertir a DataLoader
    train_data_subset = create_batch(X_train_subset, edge_index, mask_train_subset, y_train_subset, device)
    subset_train_loader = DataLoader([train_data_subset], batch_size=1, shuffle=False)

    train_data = create_batch(X_train, edge_index, mask_train, y_train, device)
    train_loader = DataLoader([train_data], batch_size = 1, shuffle = False)
    del X_train_subset,y_train_subset,mask_train_subset,train_data

    gc.collect()
    torch.cuda.empty_cache()
    print('OPTUNA FASE 1')
    # Primera fase de optimización (búsqueda rápida)
    study_phase1 = optuna.create_study(direction='minimize',study_name = 'FASE 1')

    # study_phase1 = optuna.create_study(direction='maximize',study_name = 'FASE 1')

    study_phase1.optimize(modeloptunaF1_wrapper(subset_train_loader = subset_train_loader,
                                                val_loader = val_loader,
                                                out = out, shapeNF = shapeNF,
                                                timewindow = timewindow,device = device,
                                                num_epochs=num_epochs[0],weights=loss_weights_), n_trials=n_trials[0])

    # Guardar los mejores hiperparámetros de la fase 1
    best_params = study_phase1.best_trial.params
    torch.cuda.empty_cache()
    del study_phase1

    print("Mejores parámetros de la fase 1:", best_params)
    if phases:
        print('OPTUNA FASE 2')
        # Segunda fase de optimización (refinada)

        subset_size = int(subset_prop[1] * len(X_train))  # Subset del X%
        start_idx = torch.randint(low = 0, high = int((0.99 - subset_prop[1]) * len(X_train)), size = (1,))

        # Crear el subset correlativo
        X_train_subset = X_train[start_idx:start_idx + subset_size]
        y_train_subset = y_train[start_idx:start_idx + subset_size]
        mask_train_subset = mask_train[start_idx:start_idx + subset_size]

        # Convertir a DataLoader
        train_data_subset2 = create_batch(X_train_subset, edge_index, mask_train_subset, y_train_subset, device)
        subset_train_loader2 = DataLoader([train_data_subset2], batch_size = 1, shuffle = False)

        del X_train_subset, y_train_subset, mask_train_subset, train_data_subset2



        study_phase2 = optuna.create_study(direction='minimize',study_name = 'FASE 2')
        # study_phase2 = optuna.create_study(direction='maximize',study_name = 'FASE 2')

        study_phase2.optimize(modeloptunaF2_wrapper(Opt_best = best_params, train_loader = subset_train_loader2,
                                                    val_loader = val_loader, out = out, shapeNF = shapeNF,
                                                    timewindow = timewindow, device = device,
                                                    num_epochs=num_epochs[1],weights=loss_weights_), n_trials=n_trials[0])
        torch.cuda.empty_cache()

        # Guardar los mejores hiperparámetros de la fase 2
        best_params = study_phase2.best_trial.params
        print("Mejores parámetros de la fase 2:", best_params)

        del study_phase2
    return best_params


def plot_confusion_matrix(true_labels, pred_labels, title):
    # Calcular la matriz de confusión

    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Graficar la matriz de confusión
    plt.figure(figsize = (8, 6))
    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues", cbar = False)
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()


def prepare_data(losses_df):

    mean_by_epoch = losses_df.groupby(['epoch', 'output']).mean().reset_index()
    mean_by_node = losses_df.groupby(['node', 'output']).mean().reset_index()
    return losses_df, mean_by_epoch, mean_by_node


def plot_loss_by_epoch(data, output_file = 'loss_by_epoch.png'):

    plt.figure(figsize = (12, 6))
    for output in data['output'].unique():
        output_data = data[data['output'] == output]
        plt.plot(output_data['epoch'], output_data['train_loss'], label = f'Train Loss Output {output}')
        plt.plot(output_data['epoch'], output_data['val_loss'], label = f'Val Loss Output {output}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss by Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def plot_accuracy_by_epoch(data, output_file = 'accuracy_by_epoch.png'):


    plt.figure(figsize = (12, 6))
    for output in data['output'].unique():
        output_data = data[data['output'] == output]
        plt.plot(output_data['epoch'], output_data['accuracy'], label = f'Accuracy Output {output}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy by Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def plot_loss_heatmap(data, output_file = 'loss_heatmap.png'):


    pivot_data = data.pivot(index = 'node', columns = 'output', values = 'val_loss')
    plt.figure(figsize = (15, 10))
    sns.heatmap(pivot_data, cmap = 'YlOrRd', annot = False)
    plt.xlabel('Output')
    plt.ylabel('Node')
    plt.title('Validation Loss Heatmap by Node and Output')
    plt.savefig(output_file)
    plt.close()


def plot_performance_by_node(data, metric = 'val_loss', output_file = 'performance_by_node.png'):

    plt.figure(figsize = (12, 6))
    for output in data['output'].unique():
        output_data = data[data['output'] == output]
        plt.plot(output_data['node'], output_data[metric], label = f'Output {output}')

    plt.xlabel('Node')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} by Node')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def plot_training_time(time_data, output_file = 'training_time.png'):

    plt.figure(figsize = (12, 6))
    plt.plot(time_data['epoch'], time_data['training_time'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time per Epoch')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


