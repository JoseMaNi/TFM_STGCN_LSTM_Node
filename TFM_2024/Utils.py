import os
from time import sleep
import datetime
import pickle

import dill
import zipfile
import numpy as np

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.distance import geodesic

import openmeteo_requests
import requests_cache
from retry_requests import retry

import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,OneHotEncoder

# noinspection PyTypeChecker
def unzip_csv (folder, subfolder):
    for file in os.listdir (folder):
        print('...',file)
        if file.endswith ('.zip') and not file.split ('.') [0] + '.csv' in os.listdir (os.path.join(folder,subfolder)):
            try:
                with zipfile.ZipFile (os.path.join(folder,file), 'r') as zip_ref:
                    zip_ref.extract (file.split ('.') [0] + '.csv', os.path.join(folder,subfolder))
                print ('Descomprimido el archivo: ', file)
            except Exception as e:
                print ('Error al descomprimir el archivo: ', file)
                print (e)
        elif file.endswith ('.zip') and  file.split ('.') [0] + '.csv' in os.listdir (os.path.join(folder,subfolder)):
            print(f" {file} Ya esta descomprimido en la carpeta")

    print("~~")



    # print ('+>', *os.listdir (subfolder) )

def Crear_dir (father, folder):
    try:
        os.chdir (folder)
    except FileNotFoundError:
        os.makedirs (father + '/' + folder) if not os.path.isdir (father + '/' + folder) else None

def Crear_MST (df, mstflag = True, file = 'Graph.pkl'):
    if file not in os.listdir ():
        print ('NO hay grafo en ', os.getcwd ())
        positions = df.values
        dist_matrix = distance_matrix (positions, positions)

        G = nx.Graph ()

        G.add_nodes_from (df.index)

        for i, node1 in enumerate (df.index):
            for j, node2 in enumerate (df.index):
                if i < j:
                    G.add_edge (node1, node2, weight = dist_matrix [i, j])

        mst = nx.minimum_spanning_tree (G)
        if mstflag:
            ret = mst
        else:
            ret = G

        with open (file, 'wb') as f:

            pickle.dump (ret, f)
        print ('SE GUARDA GRAFO en ', os.getcwd ())

    else:
        print ('HAY GRAFO, SE CARGA')

        with open (file, 'rb') as f:
            ret = pickle.load (f)

    return ret

def plot_graph (mst, df):
    fig, ax = plt.subplots (figsize = (20, 16))
    pos = {i: (row ['longitud'], row ['latitud']) for i, row in df.iterrows ()}
    nx.draw (mst, pos, with_labels = True, node_size = 100, node_color = 'lightblue', font_size = 10, ax = ax)
    plt.show ()

def chunker (coord, size):
    longitudes, latitudes = zip (*coord)
    for i in range (0, len (coord), size):
        yield latitudes [i:i + size], longitudes [i:i + size]

# noinspection PyTypeChecker
def Merge_Graph (G, df, on, indexcol = 'fecha', how = 'left', setindex = True, node_dependency = True,
                 drop_col = ('date',), left_index = False):
    dic = 'medidas'
    for nodo in G.nodes.keys ():
        if left_index:
            to_merge = G.nodes [nodo] [dic]
            kwargs = {'left_index': left_index,
                      'right_on': on,
                      'how': how
                      }

        else:
            to_merge = G.nodes [nodo] [dic].reset_index ()
            kwargs = {'on': on,
                      'how': how}

        if node_dependency:
            G.nodes [nodo] [dic] = pd.merge (to_merge, df [nodo],
                                             **kwargs).drop (columns = list (drop_col))
        else:
            G.nodes [nodo] [dic] = pd.merge (to_merge, df,
                                             **kwargs).drop (columns = list (drop_col))

        if setindex:
            G.nodes [nodo] [dic].set_index (indexcol, inplace = True)
            G.nodes [nodo] [dic].sort_index (inplace = True)

    return G

def abrir_csv_segun_final_concat (folder, endswith):
    df = pd.DataFrame ()
    for file in os.listdir (folder):
        # print(file)
        if file.endswith (endswith):
            df = pd.concat ([df, pd.read_csv (os.path.join (folder, file), sep = ';',
                                              encoding_errors = 'replace' )])

    return df

def plot_asignacion_cercania (gdf, Centers):
    from shapely.geometry import LineString

    lines = []
    for i, acc in gdf.iterrows ():
        clu_geom = Centers.loc [acc ['cluster'], ['longitud', 'latitud']].values
        lines.append (LineString ([acc ['geometry'], clu_geom]))

    gdf_lines = gpd.GeoDataFrame (geometry = lines)
    fig, ax = plt.subplots (figsize = (20, 20))

    gdf.plot (ax = ax, color = 'blue', marker = 'o', label = 'Accidentes', markersize = 3)
    gdf_lines.plot (ax = ax, color = 'black', linewidth = 0.5, alpha = 0.5, label = 'Asignacion_cluster')
    Centers.plot (x = 'longitud', y = 'latitud', kind = 'scatter', ax = ax, color = 'red', marker = 'o',
                  label = 'Clusters', s = 100)
    plt.legend ()

    plt.show ()

def manual_encoding (dict_map, serie):
    mapping = {}
    for grupo, rangos in dict_map.items ():
        for rango in rangos:
            mapping [rango] = grupo

    return serie.map (mapping)

def shift_in_new_col (dic, timewindow = 2, cols = None):
    if cols is None:
        cols = dic.values [0].columns

    for nodo, df in dic.items ():
        for col in cols:
            for i in range (1, timewindow + 1):
                dic [nodo] ['medidas'] [f'{col}_t-{i}'] = df ['medidas'] [col].shift (i)

    return dic

def inconsistent_ids (sens,threshold=50):
    coord_check = sens.groupby('id')[['latitud', 'longitud']].nunique()

    # Filtrar para ver dónde hay más de una coordenada por ID
    inconsistent_coords = coord_check[(coord_check['latitud'] > 1) | (coord_check['longitud'] > 1)]
    inconsistent_sens = sens.loc[inconsistent_coords.index]


    distances_by_id = inconsistent_sens.groupby('id').apply(distance_from_first).reset_index(level=0, drop = True)
    id_above_thres=distances_by_id.index[distances_by_id.distancia_a_primera > threshold].unique()

    return id_above_thres

def distance_from_first(group):
    first_coord = group[['latitud', 'longitud']].values[0]

    # Calcular distancias y crear un nuevo DataFrame
    distances = group[['latitud', 'longitud']].copy()
    distances['distancia_a_primera'] = distances.apply(
        lambda row: geodesic(first_coord, (row['latitud'], row['longitud'])).meters, axis = 1
    )

    return distances[['latitud', 'longitud', 'distancia_a_primera']]

def equal_size_graph (G,raised=True):
    try:
        series_dict = {n: data['medidas'].fecha for n,data in G.nodes.items()}
    except Exception as  e :
        print('fechas es el indice')
        print(e)
        series_dict = {n: data['medidas'].index for n, data in G.nodes.items()}

    tamanos = [index.size for n, index in series_dict.items()]

    if len(set(tamanos)) == 1:
        print("Todas las series tienen el mismo número de fechas.")


    # Si no todas tienen el mismo tamaño, proceder con el cálculo
    # Encontrar la serie con más fechas

    else:
        max_series_key = max(series_dict, key = lambda k: len(series_dict[k]))
        max_size = len(series_dict[max_series_key])

        # Comparar el tamaño de cada serie con la serie mayor
        series_faltantes = {}
        for key, fechas in series_dict.items():
            if len(fechas) < max_size:
                print(key,' : ',len(fechas), 'vs', max_size)
        if raised:
            raise ValueError("Los indices cambian.")

def cyclic_encoder(df_, col):
    df=df_.copy()
    max_val = df[col].max()  # Hallar el valor máximo de la columna
    # Crear las nuevas columnas 'sin' y 'cos' usando el valor máximo
    df[f'{col}_sin'] = (np.sin(2 * np.pi * df[col] / max_val).astype(np.float32))
    df[f'{col}_cos'] = (np.cos(2 * np.pi * df[col] / max_val).astype(np.float32))
    df.drop(columns=col,inplace=True)
    return df
