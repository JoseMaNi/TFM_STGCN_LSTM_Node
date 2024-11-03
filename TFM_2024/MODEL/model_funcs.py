
import time
import pandas as pd
import gc
from sklearn.metrics import f1_score, precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F_
from torch_geometric.nn import GCNConv

from tensorflow.python.layers.core import dropout

import torch.optim as optim

from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from torch.amp import  autocast, GradScaler

from CONSTANTS import *

from MODEL.preprocessing_G import *

from main import main


def prepare_dataset (device):

    G=main()

    FEATURE_COLS = G.nodes[0]['medidas'].columns.drop(DROP_COLS+LABEL_COLS)
    T0 = G.nodes[0]['medidas'].shape[0]
    F0 = G.nodes[0]['medidas'].shape[1]
    N0 = len(G.nodes())


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
    # X_y_test = torch.cat((X_test, y_test), dim = 2)

    # Guardamos datos test para graficar después
    print('------------')
    # Creación de los DataLoaders
    train_loader = DataLoader([train_data], batch_size = batch_size, shuffle = False)#,pin_memory=True)
    val_loader = DataLoader([val_data], batch_size = batch_size, shuffle = False)#,pin_memory=True)
    test_loader = DataLoader([test_data], batch_size = batch_size, shuffle = False)#,pin_memory=True)
    torch.save(test_loader,TEST_DATA)


    del X_val, X_test, y_val, y_test, mask_val, mask_test
    del train_data,val_data,test_data
    torch.cuda.empty_cache()
    gc.collect()

    T, N, F = X_train.shape

    return (T, N, F), (X_train,y_train, mask_train, edge_index) , (train_loader, val_loader,test_loader)

def generate_sequences_node(x, masks, sequence_length = 5):

    x = torch.as_tensor(x)
    masks = torch.as_tensor(masks, dtype = torch.bool)


    # Verificamos las dimensiones de entrada
    if x.dim() != 2:
        raise ValueError(f"x debe ser un tensor 2D, pero tiene {x.dim()} dimensiones")

    if masks.dim() == 1:
        masks = masks.unsqueeze(1)
    elif masks.dim() > 2:
        raise ValueError(f"masks debe ser un tensor 1D o 2D, pero tiene {masks.dim()} dimensiones")

    num_rows, num_cols = x.shape
    device = x.device

    if masks.shape[0] != num_rows:
        raise ValueError(
            f"El número de filas en masks ({masks.shape[0]}) debe coincidir con el número de filas en x ({num_rows})")

    indices = torch.arange(num_rows, device = device).unsqueeze(1).expand(-1, sequence_length)
    offsets = torch.arange(sequence_length - 1, -1, -1, device = device).unsqueeze(0).expand(num_rows, -1)
    sequence_indices = indices - offsets

    mask_expanded = masks.expand(-1, sequence_length)
    masked_indices = torch.where(mask_expanded, sequence_indices, 0)
    valid_indices = torch.clamp(masked_indices, min = 0, max = num_rows - 1)


    sequences = x[valid_indices]
    final_mask = (masked_indices >= 0) & (masked_indices < num_rows) & mask_expanded
    sequences = torch.where(final_mask.unsqueeze(-1), sequences, torch.zeros_like(sequences))


    return sequences[masks.any(dim = 1).squeeze()]

def create_batch(features, edge_index, masks, labels, device):
    T, N, F = features.shape
    _, E = edge_index.shape
    data_list = []
    # print(labels.shape)

    for t in range(T):
        data = Data(
            x = features[t],  # [N, F]
            edge_index = edge_index,
            mask = masks[t],  # [N]
            y = labels[t]  # [N, num_labels]
        )
        data_list.append(data)

    batch=Batch.from_data_list(data_list).to(device)

    batch.x=batch.x.view(features.shape)
    batch.mask=batch.mask.view(masks.shape)
    batch.y = batch.y.view(labels.shape)


    batch.edge_index = edge_index.unsqueeze(0).repeat(T, 1, 1)

    batch.batch_size = T

    return batch

def train_step(model, optimizer, batch, criterions, loss_weights, device,scaler):
    batch = batch.to(device)

    if torch.all(batch.mask == 0):
        print('Advertencia: Batch con máscara completamente en cero encontrado. Saltando este batch.')
        return None, None

    optimizer.zero_grad()

    # Usar autocast para habilitar precisión mixta
    with autocast(device.type):
        try:
            print('Forward pass')
            outputs, targets = model(batch)
            # Calcular la pérdida usando precisión mixta
            loss, mean_loss_by_output = combined_loss(outputs, targets, criterions, loss_weights)

        except RuntimeError as e:
            print(f'Error en el cálculo de la pérdida o forward pass: {str(e)}')
            return None, None

    try:
        print('Backward pass')
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        return loss.item(), mean_loss_by_output

    except RuntimeError as e:
        print(f'Error en la backpropagation: {str(e)}')
        return None, None

def train_model(model, train_loader, val_loader,lr,weight_decay, criterions, num_epochs, loss_weights, device,
                metrics = True,scheduler_factor=0.5,early_stopping=(6,0.5)):
    best_model_wts=None
    patience, min_delta = early_stopping
    patience_sch=patience//3
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr = lr,weight_decay = weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=patience_sch)
    model = model.to(device)

    all_metrics_df = pd.DataFrame(columns = ['epoch', 'output', 'train_loss', 'val_loss', 'val_precision','val_f1_score'])
    time_df = pd.DataFrame(columns = ['epoch', 'training_time'])
    # classification_report_df = pd.DataFrame(columns=['epoch', 'node', 'output', 'label', 'precision', 'recall', 'f1-score'])
    val_loss, counter, best_loss = 0, 0, -1000
    print('**TRAIN**')
    for epoch in range(num_epochs):
        print('** EPOCH', epoch + 1, '**')
        time_start = time.time()
        model.train()
        # Bucle de entrenamiento principal
        total_loss = 0
        mean_losses_by_output_train = []
        valid_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            loss, mean_loss_by_output = train_step(model, optimizer, batch, criterions, loss_weights, device,scaler)

            if loss is not None:
                total_loss += loss
                mean_losses_by_output_train.append(mean_loss_by_output)
                valid_batches += 1
            batch = batch.to('cpu')
            # torch.cuda.empty_cache()

        # Calcular la pérdida promedio solo para los batches válidos
        average_loss = total_loss / valid_batches if valid_batches > 0 else 0
        # train_loader = train_loader.to('cpu')
        torch.cuda.empty_cache()
        print(f"Pérdida promedio de entrenamiento: {average_loss:.4f}")
        print(f"Batches válidos procesados: {valid_batches}/{len(train_loader)}")

        # Evaluación
        val_loss, val_acc, val_metrics_by_output = evaluate_model(model,
                                                                  val_loader,
                                                                  criterions,
                                                                  loss_weights,
                                                                  device)

        scheduler.step(val_loss)

        epoch_time = (time.time() - time_start) / 60
        if metrics:
            all_metrics_df, time_df = store_metrics(epoch, mean_loss_by_output,
                val_loss, val_metrics_by_output,
                all_metrics_df,
                time_df, epoch_time)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Tiempo transcurrido: {epoch_time}')
        print(
            f'Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {[f"{acc:.4f}" for acc in val_acc]}')

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            counter = 0
            best_model_wts = model.state_dict()
        else:
            print(f'Best loss {best_loss} | Val Loss {val_loss} |  min_delta {min_delta}')
            counter += 1
            print(f'Paciencia Early Stopping: {counter}/{patience}')

        # Detener si no hay mejora después de 'patience' épocas
        if counter >= patience:
            print("Early stopping activado")
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return all_metrics_df, time_df

def evaluate_model__(model, val_loader, criterions,loss_weights, device,testing_flag=False):
    print('--EVAL--')
    model.eval()
    total_loss = 0
    total_acc = [0] * len(criterions)
    total_samples = [0] * len(criterions)

    f1_scores_by_output = []
    precisions_by_output = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            outputs, targets = model(batch)

            # Calcular la pérdida y obtener las pérdidas por salida
            loss, mean_loss_by_output = combined_loss(outputs, targets, criterions,loss_weights)

            total_loss += loss.item()


            for i in range(len(outputs)):  # Para cada salida
                all_preds, all_labels = [], []
                for node in range(targets.shape[1]):  # Para cada nodo

                    preds = outputs[i][:, node, :].argmax(dim=1).cpu().numpy()  # Predicciones
                    labels = targets[:, node, i].cpu().numpy()  # Etiquetas reales

                    all_preds.extend(preds)
                    all_labels.extend(labels)

                    # Contar las predicciones correctas y sumar al total
                    correct = (preds == labels).sum().item()
                    total_acc[i] += correct
                    total_samples[i] += len(labels)

                # Calcular métricas finales para la salida i
                precision = precision_score(all_labels, all_preds, average = 'macro', zero_division = 0)
                f1_score_final = f1_score(all_labels, all_preds, average = 'macro', zero_division = 0)

                precisions_by_output.append(precision)
                f1_scores_by_output.append(f1_score_final)
            batch = batch.to('cpu')
    torch.cuda.empty_cache()

    # Calcular la pérdida media por salida (promediando entre nodos)

    avg_loss = total_loss / len(val_loader)
    avg_acc = [total_acc[i] / total_samples[i] for i in range(len(total_acc))]
    metrics = {'precisions': precisions_by_output, 'f1_scores': f1_scores_by_output}

    if  testing_flag:
        matrix = pd.DataFrame({'Predictions': preds, 'Labels': labels})
        test_metrics = []
        num_outputs = len(metrics['precisions'])

        for output in range(num_outputs):
            test_metrics.append({
                'output': output,
                'test_loss': avg_loss,
                'test_precision': metrics['precisions'][output],
                'test_f1_score': metrics['f1_scores'][output],
            })

        test_metrics_df = pd.DataFrame(test_metrics)

        test_metrics_df.to_csv(TEST_METRICS_FILE, index = False)

        matrix.to_csv(PREDS_LABELS_FILE, index = False)
    else:
        return avg_loss, avg_acc ,metrics

def evaluate_model(model, val_loader, criterions, loss_weights, device, testing_flag = False):
    print('--EVAL--')
    model.eval()
    total_loss = 0
    total_acc = torch.zeros(len(criterions), device = device)
    total_samples = torch.zeros(len(criterions), device = device)

    f1_scores_by_output = []
    precisions_by_output = []

    # Almacenar todas las predicciones y etiquetas de forma tensorial
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs, targets = model(batch)

            loss, mean_loss_by_output = combined_loss(outputs, targets, criterions, loss_weights)
            total_loss += loss.item()

            # Convertir outputs a predicciones usando argmax de forma tensorial
            preds_tensor = torch.stack([out.argmax(dim = -1) for out in outputs],dim = -1)  #batch, nodos, num_salidas
            labels_tensor = torch.stack([targets[:, :, i] for i in range(len(outputs))], dim = -1)

            # Acumular las predicciones y etiquetas para análisis por lotes
            all_predictions.append(preds_tensor.cpu())
            all_labels.append(labels_tensor.cpu())

            # Calcular precisión y métricas
            for i in range(len(outputs)):
                correct = (preds_tensor[:, :, i] == labels_tensor[:, :, i]).sum().item()
                total_acc[i] += correct
                total_samples[i] += preds_tensor.shape[0] * preds_tensor.shape[1]  # total por batch y nodos

                # Calcular precisión y F1
                all_preds = preds_tensor[:, :, i].flatten().cpu().numpy()
                all_labels_flat = labels_tensor[:, :, i].flatten().cpu().numpy()

                precision = precision_score(all_labels_flat, all_preds, average = 'macro', zero_division = 0)
                f1_score_final = f1_score(all_labels_flat, all_preds, average = 'macro', zero_division = 0)

                precisions_by_output.append(precision)
                f1_scores_by_output.append(f1_score_final)

            batch = batch.to('cpu')

    all_predictions = torch.cat(all_predictions, dim = 0)  # (total_samples, nodos, num_salidas)
    all_labels = torch.cat(all_labels, dim = 0)  # (total_samples, nodos, num_salidas)

    result_df = pd.DataFrame({
        'Predictions': all_predictions.view(-1, all_predictions.shape[-1]).tolist(),
        'Labels': all_labels.view(-1, all_labels.shape[-1]).tolist()
    })

    # Calcular la pérdida media por salida
    avg_loss = total_loss / len(val_loader)
    avg_acc = (total_acc / total_samples).tolist()
    metrics = {'precisions': precisions_by_output, 'f1_scores': f1_scores_by_output}

    # Guardar si estamos en el modo de prueba
    if testing_flag:
        result_df.to_csv(PREDS_LABELS_FILE, index = False)

        test_metrics = []
        num_outputs = len(metrics['precisions'])

        for output in range(num_outputs):
            test_metrics.append({
                'output': output,
                'test_loss': avg_loss,
                'test_precision': metrics['precisions'][output],
                'test_f1_score': metrics['f1_scores'][output],
            })

        test_metrics_df = pd.DataFrame(test_metrics)
        test_metrics_df.to_csv(TEST_METRICS_FILE, index = False)

    else:
        return avg_loss, avg_acc, metrics

def combined_loss(outputs, targets, criterions,weights):
    total_loss = 0

    losses_by_output = []

    # Iterar sobre cada una de las salidas por nodo
    for i, (criterion,weight) in enumerate(zip(criterions,weights)):
        losses_per_node = []
        for node in range(targets.shape[1]):
            target = targets[:, node, i].long()
            output = outputs[i][:, node, :]

            loss = criterion(output, target) * weight # La pérdida es ponderada, cada salida posee un peso
            losses_per_node.append(loss.item())

            total_loss += loss
        del target, output, loss
        gc.collect()
        # torch.cuda.empty_cache()


        losses_by_output.append(sum(losses_per_node) / len(losses_per_node))

    return total_loss,  losses_by_output

def store_metrics(epoch, train_losses, val_loss, val_metrics_by_output, all_metrics_df, time_df, epoch_time):
    new_metrics = []

    num_outputs = len(train_losses)

    for output in range(num_outputs):
        new_metrics.append({
            'epoch': epoch + 1,
            'output': output,
            'train_loss': train_losses[output],
            'val_loss': val_loss,
            'val_precision': val_metrics_by_output['precisions'][output],
            'val_f1_score': val_metrics_by_output['f1_scores'][output],
        })

    # Crear nuevo DataFrame con las métricas
    new_metrics_df = pd.DataFrame(new_metrics)
    new_time_df = pd.DataFrame({'epoch': [epoch + 1], 'training_time': [epoch_time]})

    # Concatenar solo si hay nuevos datos
    if not new_metrics_df.empty:
        all_metrics_df = pd.concat([all_metrics_df, new_metrics_df], ignore_index = True)
    time_df = pd.concat([time_df, new_time_df], ignore_index = True)

    # Limpiar variables no necesarias
    del new_metrics, new_metrics_df, new_time_df

    return all_metrics_df, time_df