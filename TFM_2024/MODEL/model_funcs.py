import torch
import torch.nn as nn
import torch.nn.functional as F_
from torch_geometric.nn import GCNConv
import torch.optim as optim
from tensorflow.python.layers.core import dropout

from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
import time
import pandas as pd
import gc
from sklearn.metrics import f1_score, precision_score
from torch.amp import autocast, GradScaler
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
    T, N, F_ = features.shape
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

def train_model_classic(model, train_loader, val_loader, criterions, optimizer, num_epochs,loss_weigths, device,metrics=True):
    model = model.to(device)

    all_metrics_df = pd.DataFrame(columns=['epoch', 'node', 'output', 'train_loss', 'val_loss', 'accuracy'])
    time_df = pd.DataFrame(columns=['epoch', 'training_time'])
    # classification_report_df = pd.DataFrame(columns=['epoch', 'node', 'output', 'label', 'precision', 'recall', 'f1-score'])

    print('**TRAIN**')
    for epoch in range(num_epochs):
        print('** EPOCH', epoch + 1, '**')
        time_start = time.time()
        model.train()
        total_loss = 0
        # losses_by_node_and_output_train = []
        mean_losses_by_output_train = []  # Almacenar la pérdida media por salida

        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)

            outputs, targets = model(batch)

            loss, mean_loss_by_output = combined_loss(outputs, targets, criterions,loss_weigths)
            # losses_by_node_and_output_train.append(losses_by_node_and_output)
            mean_losses_by_output_train.append(mean_loss_by_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()



        (val_loss, val_acc,val_metrics_by_output) = evaluate_model(model,
                                                           val_loader,
                                                           criterions,
                                                           loss_weigths ,
                                                           device)

        epoch_time = (time.time() - time_start) / 60
        if metrics:
            all_metrics_df, time_df = store_metrics(
                                                    epoch,
                                                    mean_losses_by_output_train[-1],
                                                    val_loss,val_metrics_by_output,
                                                    all_metrics_df,
                                                    time_df, epoch_time)
        torch.cuda.empty_cache()

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Tiempo transcurrido: {epoch_time}')
        print(f'Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {[f"{acc:.4f}" for acc in val_acc]}')


    return all_metrics_df, time_df


def train_step(model, optimizer, batch, criterions, loss_weights, device,scaler):
    batch = batch.to(device)

    if torch.all(batch.mask == 0):
        print("Advertencia: Batch con máscara completamente en cero encontrado. Saltando este batch.")
        return None, None

    optimizer.zero_grad()

    # Usar autocast para habilitar precisión mixta
    with autocast(device.type):
        try:
            # Forward pass
            outputs, targets = model(batch)

            print('1')
            # Calcular la pérdida usando precisión mixta
            loss, mean_loss_by_output = combined_loss(outputs, targets, criterions, loss_weights)
            print('2')

        except RuntimeError as e:
            print(f"Error en el cálculo de la pérdida o forward pass: {str(e)}")
            return None, None

    try:
        # Backward pass usando GradScaler
        scaler.scale(loss).backward()
        print('3')

        # Actualizar los parámetros con GradScaler
        scaler.step(optimizer)
        scaler.update()

        return loss.item(), mean_loss_by_output

    except RuntimeError as e:
        print(f"Error en la backpropagation: {str(e)}")
        return None, None


def train_model(model, train_loader, val_loader,lr,weight_decay, criterions, num_epochs, loss_weights, device, metrics = True,scheduler_factor=0.5,early_stopping=[6,0.5]):
    best_model_wts=None
    patience, min_delta = early_stopping
    patience_sch=patience//3
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr = lr,weight_decay = weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=patience_sch, verbose=True)
    model = model.to(device)
    all_metrics_df = pd.DataFrame(columns = ['epoch', 'node', 'output', 'train_loss', 'val_loss', 'accuracy'])
    time_df = pd.DataFrame(columns = ['epoch', 'training_time'])
    # classification_report_df = pd.DataFrame(columns=['epoch', 'node', 'output', 'label', 'precision', 'recall', 'f1-score'])

    print('**TRAIN**')
    for epoch in range(num_epochs):
        print('** EPOCH', epoch + 1, '**')
        time_start = time.time()
        model.train()
        # Bucle de entrenamiento principal
        total_loss = 0
        mean_losses_by_output_train = []
        valid_batches = 0

        loss, counter,best_loss = 0,0,-1000

        for batch in train_loader:
            batch = batch.to(device)
            loss, mean_loss_by_output = train_step(model, optimizer, batch, criterions, loss_weights, device,scaler)

            if loss is not None:
                total_loss += loss
                mean_losses_by_output_train.append(mean_loss_by_output)
                valid_batches += 1
            batch = batch.to('cpu')
            torch.cuda.empty_cache()

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
            all_metrics_df, time_df = store_metrics(
                epoch,
                mean_losses_by_output_train[-1],
                val_loss, val_metrics_by_output,
                all_metrics_df,
                time_df, epoch_time)
        torch.cuda.empty_cache()

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Tiempo transcurrido: {epoch_time}')
        print(
            f'Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {[f"{acc:.4f}" for acc in val_acc]}')



        if loss < best_loss - min_delta:
            best_loss = loss
            counter = 0

            best_model_wts = model.state_dict()
        else:
            counter += 1

        # Detener si no hay mejora después de 'patience' épocas
        if counter >= patience:
            print("Early stopping activado")
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return all_metrics_df, time_df




def evaluate_model(model, val_loader, criterions,loss_weights, device):
    print('--EVAL--')
    model.eval()
    total_loss = 0
    total_acc = [0] * len(criterions)
    total_samples = [0] * len(criterions)
    # losses_by_node_and_output_val = []
    # classification_reports = []

    f1_scores_by_output = []
    precisions_by_output = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            outputs, targets = model(batch)

            # Calcular la pérdida y obtener las pérdidas por nodo y salida
            loss, mean_loss_by_output = combined_loss(outputs, targets, criterions,loss_weights)

            # losses_by_node_and_output_val.append(losses_by_node_and_output)
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

    return avg_loss, avg_acc ,{'precisions': precisions_by_output, 'f1_scores': f1_scores_by_output}

def combined_loss(outputs, targets, criterions,weights):
    total_loss = 0
    # losses_by_node_and_output = []
    losses_by_output = []

    # Iterar sobre cada una de las salidas por nodo
    for i, (criterion,weight) in enumerate(zip(criterions,weights)):
        losses_per_node = []
        for node in range(targets.shape[1]):
            target = targets[:, node, i].long()
            output = outputs[i][:, node, :]

            loss = criterion(output, target) * weight
            losses_per_node.append(loss.item())

            total_loss += loss
        del target, output, loss
        gc.collect()
        torch.cuda.empty_cache()


        # losses_by_node_and_output.append(losses_per_node)
        losses_by_output.append(sum(losses_per_node) / len(losses_per_node))

    return total_loss,  losses_by_output

# def store_metrics_and_reports(epoch, losses_train, losses_val, val_acc, classification_reports, all_metrics_df,
#                               classification_report_df, time_df, epoch_time):
#     new_metrics = []
#     new_reports = []
#
#     num_outputs = len(losses_val)
#     num_nodes = len(losses_val[0]) if num_outputs > 0 else 0
#
#     for output in range(num_outputs):
#         for node in range(num_nodes):
#             new_metrics.append({
#                 'epoch': epoch + 1,
#                 'node': node,
#                 'output': output,
#                 'train_loss': losses_train[output][node],
#                 'val_loss': losses_val[output][node],
#                 'accuracy': val_acc[output]
#             })
#
#             # Agregar resultados del classification report
#             if classification_reports and len(classification_reports) > 0:
#                 report = classification_reports[0][output][node]
#                 for label, metrics in report.items():
#                     if isinstance(metrics, dict):  # Excluir métricas agregadas
#                         new_reports.append({
#                             'epoch': epoch + 1,
#                             'node': node,
#                             'output': output,
#                             'label': label,
#                             'precision': metrics['precision'],
#                             'recall': metrics['recall'],
#                             'f1-score': metrics['f1-score']
#                         })
#
#     # Crear nuevos DataFrames con los datos de esta época
#     new_metrics_df = pd.DataFrame(new_metrics)
#     new_reports_df = pd.DataFrame(new_reports)
#     new_time_df = pd.DataFrame({'epoch': [epoch + 1], 'training_time': [epoch_time]})
#
#     # Concatenar solo si hay nuevos datos
#     if not new_metrics_df.empty:
#         all_metrics_df = pd.concat([all_metrics_df, new_metrics_df], ignore_index=True)
#     if not new_reports_df.empty:
#         classification_report_df = pd.concat([classification_report_df, new_reports_df], ignore_index=True)
#     time_df = pd.concat([time_df, new_time_df], ignore_index=True)
#
#     return all_metrics_df, classification_report_df, time_df
#

def store_metrics(epoch, train_losses, val_loss, val_metrics_by_output, all_metrics_df, time_df, epoch_time):
    new_metrics = []

    num_outputs = len(train_losses)

    for output in range(num_outputs):
        new_metrics.append({
            'epoch': epoch + 1,
            'output': output,
            'train_loss': train_losses[output],
            'val_loss': val_loss,
            'precision': val_metrics_by_output['precisions'][output],
            'f1_score': val_metrics_by_output['f1_scores'][output],
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