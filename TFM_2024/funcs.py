import numpy as np
import pandas as pd

from Utils import *

pd.set_option('future.no_silent_downcasting', True)
def llenar_grafo_medidas (G, medidas, indexrange):
    shape = indexrange.shape
    for name, data in medidas.items ():

        for cluster in data ['cluster'].unique ():
            medidas_cluster = data.loc [data ['cluster'] == cluster]

            if 'medidas' not in G.nodes [cluster]:
                G.nodes [cluster] ['medidas'] = pd.DataFrame ()

            G.nodes [cluster] ['medidas'] = pd.concat ([G.nodes [cluster] ['medidas'], medidas_cluster])

    for nodo, data in G.nodes.items ():
        print (nodo)

        G.nodes [nodo] ['medidas'].sort_values (by = 'fecha', inplace = True)
        G.nodes [nodo] ['medidas'] = (data ['medidas'].set_index ('fecha')
                                      .reindex (indexrange)
                                      .ffill ()
                                      .bfill ()
                                      )
        G.nodes [nodo] ['medidas'].index.name = 'fecha'
        print (shape, '----', G.nodes [nodo] ['medidas'].shape)

        # medidas[name].drop(index=cluster, level='cluster', inplace=True)

    return G


def Calendario (file, G):
    Cal = pd.read_csv (file, sep = ';', index_col = 0).iloc [:, :-2]
    Cal.index = pd.to_datetime (Cal.index, format = '%d/%m/%Y')
    Cal.loc [Cal ['Dia_semana'] == 'miércoles', 'Dia_semana'] = 'miercoles'
    Cal.loc [Cal.Dia_semana == 'sábado', 'Dia_semana'] = 'sabado'
    Cal.loc [Cal.laborable_festivo == 'Festivo', 'laborable_festivo'] = 'festivo'
    Cal.loc [Cal.TipoFestivo == 'Fiesta de la Comunidad de Madrid', 'TipoFestivo'] = 'Festivo de la Comunidad de Madrid'
    Cal.loc [
        Cal.TipoFestivo == 'Festivo de la comunidad de Madrid', 'TipoFestivo'] = 'Festivo de la Comunidad de Madrid'
    Cal.loc [Cal.TipoFestivo == 'Fiesta Comunidad de Madrid', 'TipoFestivo'] = 'Festivo de la Comunidad de Madrid'
    Cal.loc [
        Cal.TipoFestivo == 'traslado de la Fiesta de la Comunidad de Madrid', 'TipoFestivo'] = ('Festivo de la '
                                                                                                'Comunidad de Madrid')
    Cal.loc [Cal.TipoFestivo == 'Fiesta local', 'TipoFestivo'] = 'Festivo local'
    Cal.loc [Cal.TipoFestivo == 'Festivo local de la ciudad de Madrid', 'TipoFestivo'] = 'Festivo local'
    Cal.loc [Cal.TipoFestivo == 'Fiesta nacional', 'TipoFestivo'] = 'Festivo nacional'

    to_OH=['laborable_festivo','TipoFestivo']
    to_Label=['Dia_semana']

    Cal.reset_index (inplace = True)
    Cal ['day'] = Cal ['Dia'].dt.day
    Cal ['month'] = Cal ['Dia'].dt.month
    Cal ['year'] = Cal ['Dia'].dt.year
    LE=LabelEncoder()
    for col in to_Label:
        Cal[col]=LE.fit_transform (Cal [col])
    Cal = cyclic_encoder(Cal,'Dia_semana')
    Cal = pd.get_dummies(Cal,columns = to_OH)

    to_drop=['Festividad','Dia']
    Cal.drop (columns = to_drop, inplace = True)


    print (Cal.head (2))


    return Merge_Graph (G, Cal, on = ['year', 'month', 'day'], how = 'left', node_dependency = False,
                        setindex = True,drop_col = [])


def Meteo_req (Coords, datemin, datemax):
    cache_session = requests_cache.CachedSession ('.cache', expire_after = -1)
    retry_session = retry (cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client (session = retry_session)

    url = 'https://archive-api.open-meteo.com/v1/archive'
    all_results = []
    batches = list (chunker (Coords.values.tolist (), 10))
    datemin = pd.to_datetime (datemin) - pd.Timedelta (days = 1)
    datemax = pd.to_datetime (datemax) + pd.Timedelta (days = 2)

    print ('_________')
    print (f'Desde {datemin.strftime ('%Y-%m-%d')} -- Hasta {datemax.strftime ('%Y-%m-%d')}')
    for i, batch in enumerate (batches):
        try:
            print (f'Batch  {i + 1}  de {len (batches)}')

            params = {
                'latitude': batch [0],
                'longitude': batch [1],
                'start_date': datemin.strftime ('%Y-%m-%d'),
                'end_date': datemax.strftime ('%Y-%m-%d'),
                'hourly': ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'rain', 'is_day',
                           'sunshine_duration'],
                'timezone': 'Europe/Berlin',
            }
            all_results.extend (openmeteo.weather_api (url, params = params))

        except Exception as e:
            # Captura cualquier excepción y muestra su tipo y mensaje
            print (f'Se produjo un error: {type (e).__name__}')
            # print(f'Mensaje del error: {e}')
            if type (e).__name__ == 'OpenMeteoRequestsError':
                print ('Superado el limite de peticiones')
                print (f'Esperando 90secs \nVamos por el batch  {i + 1}  de {len (batches)}')

                sleep (90)
                params = {
                    'latitude': batch [0],
                    'longitude': batch [1],
                    'start_date': datemin.strftime ('%Y-%m-%d'),
                    'end_date': datemax.strftime ('%Y-%m-%d'),
                    'hourly': ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'rain',
                               'is_day', 'sunshine_duration'],
                    'timezone': 'Europe/Berlin',
                }
                print('Se continua.....')
                all_results.extend (openmeteo.weather_api (url, params = params))
                print('@@@')


    print ('_________')

    return all_results



def Meteo_parse (Responses):
    hourly_dict = {}

    for nodo, response in enumerate (Responses):
        # print(f'Cl {nodo}: {np.round(response.Latitude(),5) }°N {np.round(response.Longitude(),5)}°E')

        hourly = response.Hourly ()
        hourly_temperature_2m = hourly.Variables (0).ValuesAsNumpy ()
        hourly_relative_humidity_2m = hourly.Variables (1).ValuesAsNumpy ()
        hourly_dew_point_2m = hourly.Variables (2).ValuesAsNumpy ()
        hourly_precipitation = hourly.Variables (3).ValuesAsNumpy ()
        hourly_rain = hourly.Variables (4).ValuesAsNumpy ()
        hourly_is_day = hourly.Variables (5).ValuesAsNumpy ()
        hourly_sunshine_duration = hourly.Variables (6).ValuesAsNumpy ()

        hourly_batch = {'date': pd.date_range (
            start = pd.to_datetime (hourly.Time (), unit = 's', utc = True),
            end = pd.to_datetime (hourly.TimeEnd (), unit = 's', utc = True),
            freq = pd.Timedelta (seconds = hourly.Interval ()),
            inclusive = 'left'
        )}

        hourly_batch ['temperature_2m'] = hourly_temperature_2m
        hourly_batch ['relative_humidity_2m'] = hourly_relative_humidity_2m
        hourly_batch ['dew_point_2m'] = hourly_dew_point_2m
        hourly_batch ['precipitation'] = hourly_precipitation
        hourly_batch ['rain'] = hourly_rain
        hourly_batch ['is_day'] = hourly_is_day
        hourly_batch ['sunshine_duration'] = hourly_sunshine_duration

        hourly_batch = pd.DataFrame (hourly_batch)

        hourly_batch ['date'] = pd.to_datetime (hourly_batch ['date'])
        hourly_batch ['year'] = hourly_batch ['date'].dt.year
        hourly_batch ['month'] = hourly_batch ['date'].dt.month
        hourly_batch ['day'] = hourly_batch ['date'].dt.day
        hourly_batch ['hour'] = hourly_batch ['date'].dt.hour

        # pd.merge(G.nodes[nodo]['medidas'].reset_index(),Cal,on=['year','month','day'],how='left').drop(columns=[
        # 'Dia'])
        hourly_dict [nodo] = hourly_batch

    return hourly_dict

def Accidentes (folder, Coords, gravedad,endswith='idad.csv',aggregate=False):

    Acc = abrir_csv_segun_final_concat (folder, endswith)

    Acc ['fecha'] = Acc ['fecha'] + ' ' + Acc ['hora']
    col_drop = ['localizacion', 'numero', 'cod_distrito', 'distrito', 'tipo_accidente',
                'tipo_vehiculo', 'tipo_persona', 'lesividad', 'positiva_alcohol',
                'positiva_droga', 'hora', 'sexo']
    col_drop +=[c for c in Acc.columns if c.startswith('estado_meteorol') ]
    print(col_drop)
    Acc.drop (col_drop, axis = 1, inplace = True)

    Acc.loc [(Acc ['cod_lesividad'].isnull ()) | (Acc ['cod_lesividad'] == 77), 'cod_lesividad'] = 14
    Acc.fecha = pd.to_datetime (Acc.fecha, format = '%d/%m/%Y %H:%M:%S')

    # print('Accidentes antes', Acc)
    Acc ['coordenada_x_utm'].dropna (inplace = True)
    Acc ['coordenada_y_utm'].dropna (inplace = True)
    Acc=Acc.loc[(Acc['coordenada_x_utm']!='#¡VALOR!') | (Acc['coordenada_y_utm']!='#¡VALOR!')]
    Acc.dropna (inplace = True)
    # print('Accidentes despues', Acc)

    grupo_edad = {
        'De 0 a 17 años': ['Menor de 5 años', 'De 6 a 9 años', 'De 10 a 14 años', 'De 15 a 17 años'],
        'De 18 a 30 años': ['De 18 a 20 años', 'De 21 a 24 años', 'De 25 a 29 años'],
        'De 31 a 45 años': ['De 30 a 34 años', 'De 35 a 39 años', 'De 40 a 44 años'],
        'De 46 a 60 años': ['De 45 a 49 años', 'De 50 a 54 años', 'De 55 a 59 años'],
        'Más mayores de 60': ['De 60 a 64 años', 'De 65 a 69 años', 'De 70 a 74 años', 'Más de 74 años'],
        'Desconocido': ['Desconocido']
    }



    Acc ['rango_edad'] = manual_encoding (grupo_edad, Acc ['rango_edad'])
    Acc ['gravedad'] = manual_encoding (gravedad, Acc ['cod_lesividad'])

    Acc.drop ('cod_lesividad', axis = 1, inplace = True)

    le = LabelEncoder ()
    # Acc['sexo']=le.fit_transform(Acc['sexo'])
    Acc ['gravedad'] = le.fit_transform (Acc ['gravedad'])
    Acc ['rango_edad'] = le.fit_transform (Acc ['rango_edad'])
    Acc = Acc [Acc.coordenada_x_utm != 0]

    x = Acc ['coordenada_x_utm'].str.replace (',', '.', regex = True)
    y = Acc ['coordenada_y_utm'].str.replace (',', '.', regex = True)

    x [x.isnull ()] = Acc.loc [x.isnull (), 'coordenada_x_utm']
    y [y.isnull ()] = Acc.loc [y.isnull (), 'coordenada_y_utm']
    Acc ['coordenada_x_utm'] = pd.to_numeric (x)
    Acc ['coordenada_y_utm'] = pd.to_numeric (y)

    gdf = gpd.GeoDataFrame (Acc,
                            geometry = gpd.points_from_xy (Acc ['coordenada_x_utm'],
                                                           Acc ['coordenada_y_utm'])).set_crs (epsg = 25830)
    epsg_destino = 'EPSG:4326'
    t_prox = '15min'
    gdf = gdf.to_crs (epsg_destino)

    gdf ['fecha'] = gdf ['fecha'].dt.round (t_prox)

    gdf ['year'] = gdf.fecha.dt.year
    gdf ['month'] = gdf.fecha.dt.month
    gdf ['day'] = gdf.fecha.dt.day
    gdf ['hour'] = gdf.fecha.dt.hour
    gdf ['minute'] = gdf.fecha.dt.minute
    # print(np.array (list (zip (Coords.longitud, Coords.latitud))))
    # print('**')
    # print(np.array (list (zip (gdf.geometry.x, gdf.geometry.y))))
    tree = cKDTree (np.array (list (zip (Coords.longitud, Coords.latitud))))
    distancias, indices = tree.query (np.array (list (zip (gdf.geometry.x, gdf.geometry.y))), k = 1)
    gdf ['cluster'] = indices

    drop_col = ['coordenada_x_utm', 'coordenada_y_utm']
    gdf.drop (drop_col, axis = 1, inplace = True)

    gdf_group_list = []
    if aggregate:
        gdf_group = gdf.groupby (['cluster', 'fecha'])
        gdf_group_accidentados = (gdf_group ['num_expediente'].count ()
                                  .to_frame ()
                                  .reset_index ())
        # gdf_group_accidentes = (gdf_group ['num_expediente'].nunique ()
        #                         .to_frame ()
        #                         .reset_index ())
        gdf_group_grav = (gdf_group ['gravedad'].mean ()
                          .round (0)
                          .to_frame ()
                          .reset_index ())

        gdf_group_list.append (gdf_group_accidentados.rename (columns = {'num_expediente': 'accidentados'}))
        # gdf_group_list.append (gdf_group_accidentes.rename (columns = {'num_expediente': 'accidentes'}))
        gdf_group_list.append (gdf_group_grav)
    else:
        gdf_group = gdf.groupby (['cluster', 'fecha','num_expediente'])

        gdf_group_accidentados = (gdf_group ['num_expediente'].count ()
                                  .to_frame ()
                                  )
        gdf_group_grav = (gdf_group ['gravedad'].mean ()
                          .astype(int)
                          .to_frame ()
                          .reset_index ()
                          .drop('num_expediente',axis=1))
        gdf_group_list.append (gdf_group_grav)
        gdf_group_list.append(gdf_group_accidentados.rename(columns = {'num_expediente': 'accidentados'}))

    merge_acc = pd.DataFrame ()
    del gdf, gdf_group
    for i, df_group in enumerate (gdf_group_list):
        if i == 0:
            merge_acc = df_group
        else:
            merge_acc = pd.merge (merge_acc, df_group, on = ['cluster', 'fecha'], how = 'outer')


    merge_acc.sort_values (by = ['cluster', 'fecha'], inplace = True)


    print('ACIDENTES TRAS LIMPIAR ',merge_acc)
    return merge_acc


def prepare_acc_nodo (accidents, dates):

    accidents.set_index ('fecha', inplace = True)

    random_fechas_no_acc=dates[~dates.isin(set(accidents.index))]
    # print(accidents.iloc[0])
    random_no_acc = pd.DataFrame (
        {f : [0, 0, False] for f in random_fechas_no_acc}).T
    random_no_acc.columns = accidents.columns

    accidentsM = pd.concat ([accidents, random_no_acc]).sort_index ()
    accidentsM.index.name = 'fecha'
    accidentsM = accidentsM.reset_index ().sort_values (by = 'fecha')


    return accidentsM





def meteo (Centers, datemin, datemax):
    file = 'METEO_dict.pkl'
    if file not in os.listdir ():
        print ('NO HAY METEO')
        responses = Meteo_req (Centers, datemin, datemax)
        met = Meteo_parse (responses)

        print (met)
        with open (file, 'wb') as f:

            pickle.dump (met, f)
        print ('SE GUARDA METEO')

    else:
        print ('HAY METEO, SE CARGA')

        with open (file, 'rb') as f:
            met = pickle.load (f)

    return met


def extraer_medidas (folder, file_name, cols_v):
    medidas = {}
    # fechas_medidas = pd.DatetimeIndex([])
    ids = set()

    n = len(os.listdir(folder))

    if file_name not in os.listdir():
        print('NO HAY MEDIDAS')
        for a, file in enumerate(os.listdir(folder) ):
            print(f'{a + 1}/{n}')
            if file.endswith('.csv'):
                medida = pd.read_csv(os.path.join(folder, file),
                                     sep = ';',
                                     usecols = cols_v,
                                     encoding_errors='replace')
                medida['fecha'] = pd.to_datetime(medida['fecha'])
                # medidas[file]=medida[~medida['id'].isin(bad_ids)]
                medidas[file]=medida
                ids.update(medida.id)


        print(f'SE GUARDAN MEDIDAS en {os.getcwd()} > {file_name}......')
        with open(file_name, 'wb') as f:
            pickle.dump((medidas, ids), f)
        print('SE GUARDAN MEDIDAS')

    else:
        print('SE CARGAN MEDIDAS.....')
        with open(file_name, 'rb') as f:
            medidas, ids = pickle.load(f)
        print('SE CARGAN MEDIDAS')
        print(medidas['01-2024.csv'].head(), '\n' ,medidas['01-2024.csv'].shape)
    print('______')

    return medidas , ids


def medidas_sensores(base_folder,folders,cols,files):

    DS_folder_ubi, DS_folder_sens = folders

    unzip_csv(DS_folder_sens, DS_folder_sens)
    ubi,cols_v,cols_to_encode_ = cols
    med_file = files
    os.chdir(base_folder + '/' + DS_folder_sens)


    sens = {}

    os.chdir(base_folder + '/' + DS_folder_ubi)
    print(os.getcwd())
    for file in os.listdir():
        if file.startswith('pmed_'):
            sens[file] = pd.read_csv(file, sep = ';', usecols = ubi + ['id'])
            sens[file].set_index('id', inplace = True)
            if sens[file]['latitud'].dtype == 'object':
                sens[file]['latitud'] = (sens[file]['latitud']
                                         .str.replace('.', '', regex = False)
                                         .str.slice(0, 10)
                                         .astype(float) / 10 ** 8)
                sens[file]['longitud'] = (sens[file]['longitud']
                                          .str.replace('.', '', regex = False)
                                          .str.slice(0, 10)
                                          .astype(float) / 10 ** 8)
            # print(sens[file][['longitud', 'latitud']].head(1))
    print('___PUNTOS_MEDICION_POR_MES___')

    sens_ = pd.concat(sens.values())

    sens_unico = sens_.reset_index().drop_duplicates(subset='id',keep='first').set_index('id')

    os.chdir(base_folder)
    os.chdir(DS_folder_sens)

    medidas, ids = extraer_medidas(DS_folder_sens, med_file, cols_v)

    df = sens_unico.loc[np.intersect1d(sens_unico.index, list(ids))]

    Lencoder=LabelEncoder()
    for col in cols_to_encode_:
        df[col] = Lencoder.fit_transform(df[col])
    df.dropna(inplace = True)

    return medidas, df


def limpieza_medidas(medidas, df, cols,folder,file ):
    cols_sens, drop_cols, orden = cols
    fechas_en_nodo = {}
    print('------')
    print('Se filtran y limpian medidas defectuosas')
    if file not in os.listdir(folder):
        print('NO existen medidas limpias en ', os.path.join(folder, file))
        for name, data in medidas.items():
            # medidas[name]['cluster'] = data.id.map(df.cluster.to_dict())
            data['fecha'] = pd.to_datetime(data['fecha'])
            data['cluster'] = data.id.map(df.cluster.to_dict())

            mask = (data.error != 'N') & (data.carga == 0) & (data.intensidad == 0)
            print('_________________',(data[mask].vmed!=0).sum())
            data.loc[mask, cols_sens] = np.nan

            data = (data.ffill()
                    .drop(columns = drop_cols)
                    .groupby(orden)
                    .mean()
                    .reset_index())

            medidas[name] = data
        with open(os.path.join(folder,file), 'wb') as f:
            pickle.dump(medidas, f)
        print('SE GUARDAN MEDIDAS LIMPIAS')

    else:
        print('existen medidas limpias en ',os.path.join(folder,file))
        with open(os.path.join(folder,file), 'rb') as f:
            medidas = pickle.load(f)
        print('SE CARGAN MEDIDAS LIMPIAS')

    for name, data in medidas.items():
        cluster_dates = data.groupby('cluster')['fecha'].apply(set)
        for cluster, dates in cluster_dates.items():
            if cluster not in fechas_en_nodo:
                fechas_en_nodo[cluster] = pd.DatetimeIndex([])
            fechas_en_nodo[cluster] = fechas_en_nodo  [cluster].union(pd.DatetimeIndex(dates))


    return medidas, fechas_en_nodo


def ETL_acc (Centers, folder_acc, dates, pasos,nodos,gravedad):

    datemin,datemax, fechas_en_nodo = dates

    aggregate=True
    merge_acc = Accidentes(folder_acc, Centers, gravedad = gravedad,aggregate = aggregate)
    print(merge_acc.shape)

    merge_acc.set_index('fecha', inplace = True)
    merge_acc = merge_acc[(merge_acc.index >= datemin) & (merge_acc.index <= datemax)]

    print(merge_acc.shape)
    # ------------------------
    print('Se sacan los timesteps unicos de los accidentes...\n')
    fechas_solo_acc = merge_acc.index.unique().sort_values()

    print('Se añaden los pasos anteriores de accidentes, los no accidentes se han de filtrar por nodo...\n\n')
    filtro_solo_acc = pd.DatetimeIndex([])
    last = fechas_solo_acc

    for p in range(1, pasos + 1):
        last = last - pd.Timedelta(minutes = 15)
        filtro_solo_acc = filtro_solo_acc.union(last)

    del last
    fechas_no_acc = {}
    print('Teniendo  las fechas sin accidentes (casi todas), '
          'se eligen algunas para tener en cuenta al modelo para ese nodo.\n\n'
          'El numero depende del nodo ')
    for nodo, f in fechas_en_nodo.items():
        # print(nodo,'  ###  ',f.size)
        ##
        todo_no_acc = (~f.isin(fechas_solo_acc))
        n_acc = (merge_acc['cluster'] == nodo).sum()
        np.random.seed(106)
        fechas_no_acc[nodo] = pd.DatetimeIndex(np.random.choice(f[todo_no_acc],
                                                                n_acc, replace = False))
    # ------------------------
    acc_cluster = {}
    print('Se agregan los accidentes por nodo y se establecen columnas para el modelo\n\n')
    sumacc=0
    for nodo in nodos:
        accidents = (merge_acc.loc[merge_acc['cluster'] == nodo]).reset_index().drop('cluster', axis = 1)
        accidents['hay_accidente'] = True
        accidents = prepare_acc_nodo(accidents, fechas_no_acc[nodo])
        accidents["pad"] = True
        acc_cluster[nodo] = accidents
        sumacc+=accidents.shape[0]/2
        # print(f'En el nodo {nodo} hay {accidents.shape[0]/2} accidentes y loos mismos no-accidente')

    print(f'total de  {sumacc}  accidentes')

    filtro_no_acc = {}
    for nodo, f in fechas_no_acc.items():
        index = pd.DatetimeIndex([])
        last = f
        for p in range(1, pasos + 1):
            last = last - pd.Timedelta(minutes = 15)
            index = index.union(last)
        filtro_no_acc[nodo] = index

    ## Se añaden a los accidentes los no accidentes elegidos en los nodos,
    # pues se necesitarán para el LSTM
    filtro_total = filtro_solo_acc
    for nodo, f in filtro_no_acc.items():
        filtro_total = filtro_total.union(f)

    print('FILTRO TOTAL DE FECHAS CON ACCIDENTE (', fechas_solo_acc.size,' ) Y ',pasos,' ANTERIORES: ',filtro_total.size)
    return acc_cluster,filtro_total


def clustering(df_,clusters,cols):

    COLXY = ['longitud', 'latitud']
    print('CLUSTERING...')
    kmeans = KMeans(n_clusters = clusters, random_state = 106)
    clusters = kmeans.fit_predict(df_)

    df_['cluster'] = clusters.astype(int)

    cluster_centers = kmeans.cluster_centers_

    # df con las coordenadas de los centros
    Centers = pd.DataFrame(cluster_centers[:, 2:4], columns = COLXY)
    df_cc = pd.DataFrame(cluster_centers, columns = cols)
    print('Se crea MST desde CLUSTERS')

    G = Crear_MST(df_cc)

    return Centers, G
    # ------------------------


def completar_grafo(G,medidas,Centers, ranges,folder,file,dates):
    print('Llenar grafo con medidas')
    datemin, datemax = dates
    print('PRIMERA FECHA',datemin)
    G = llenar_grafo_medidas(G, medidas, indexrange = ranges)
    equal_size_graph(G)
    for nodo, dict_ in G.nodes.items():
        # print(dict_['medidas'].index)
        G.nodes[nodo]['medidas']['year'] = dict_['medidas'].index.year
        G.nodes[nodo]['medidas']['month'] = dict_['medidas'].index.month
        G.nodes[nodo]['medidas']['day'] = dict_['medidas'].index.day
        G.nodes[nodo]['medidas']['hour'] = dict_['medidas'].index.hour
        G.nodes[nodo]['medidas']['minute'] = dict_['medidas'].index.minute


    os.chdir(folder)
    print('DESPUES P0NER HORAS',G.nodes[0]['medidas'].head())

    G = Calendario(file, G)
    print(G.nodes[0]['medidas'].head())

    meteo_data = meteo(Centers, datemin, datemax)
    print(meteo_data[0].head())
    G = Merge_Graph(G, meteo_data, on = ['year', 'month', 'day', 'hour'], how = 'inner', setindex = True)
    print(G.nodes[0]['medidas'].head())
    print(G.nodes[0]['medidas'].shape)


    return G


def prepare_next_acc (df,to_next):#,bins_acc,num_acc):
    next_c = [d + '_next' for d in to_next]

    df = df.drop_duplicates(subset=['fecha'], keep='first')
    index_acc=df.index[df.loc[:,'hay_accidente'] == True]

    p_acc=index_acc[0]
    u_acc=index_acc[-1]
    df.loc[:,'accident_next'] = df.loc[index_acc, 'fecha'].shift(-1)

    df.loc[:u_acc, ['accident_next']] = (df.loc[:u_acc,['accident_next']].ffill().infer_objects(copy=False))
    df.loc[u_acc:, ['accident_next']]=df.loc[df.index[-1],'fecha']+pd.Timedelta(hours = 2)
    df.loc[:p_acc, ['accident_next']]=df.loc[p_acc,'fecha']

    df_sub = df.loc[index_acc,to_next].shift(-1).ffill().infer_objects(copy=False)
    df[next_c] = np.nan

    df_sub.columns = next_c
    df.loc[df_sub.index, next_c] = df_sub

    df.loc[:u_acc,next_c ] = df.loc[:u_acc,next_c ].ffill().infer_objects(copy=False)
    df.loc[u_acc:,next_c] = [1,1]
    first_v=df.loc[p_acc, to_next]

    for i,v in enumerate(first_v):
        df.loc[:p_acc, next_c[i]] = v
    # raise ValueError(df.loc[:,next_c +['accident_next']])
    df['accident_next'] = pd.to_datetime(df['accident_next'])
    df['fecha'] = pd.to_datetime(df['fecha'])

    df['accident_next'] = (df['accident_next'] - df['fecha']).dt.total_seconds() / 60
    df.loc[index_acc, 'accident_next'] = 0
    columns_to_fill = next_c + ['accident_next']

    df[columns_to_fill] = df[columns_to_fill].fillna(0)

    return df.drop(to_next,axis=1)

def transform_dataset(G,acc_cluster,aggregate=True):
    dic = 'medidas'
    indexes = []
    acccols = acc_cluster[0].columns.drop(['fecha', 'pad', 'hay_accidente'])
    print("transform_dataset")
    equal_size_graph(G)
    for nodo, dict_ in G.nodes.items():

        dict_[dic]['hour'] = dict_[dic]['hour'] + dict_[dic]['minute'] / 60 ## Decenas de minutos
        dict_[dic].drop('minute', axis = 1, inplace = True)
        to_merge = dict_[dic].reset_index()

        G.nodes[nodo][dic] = to_merge.merge(acc_cluster[nodo],
                                    on = 'fecha',
                                    how = 'left')

    for nodo, dict_ in G.nodes.items():

        dict_[dic][["pad",'hay_accidente']]=dict_[dic][["pad",'hay_accidente']].fillna(False).infer_objects(copy=False)
        dict_[dic].loc[:, acccols] = dict_[dic].loc[:, acccols].fillna(0).infer_objects(copy=False)
        G.nodes[nodo][dic] = prepare_next_acc(dict_[dic], acc_cluster[nodo].columns.drop(['hay_accidente', 'pad', 'fecha']) )
        # print('__________________________')

    return G


def scale_encode (G,bins_acc, num_acc):

    COLS_G_STD = ['intensidad', 'ocupacion', 'carga', 'vmed',
                  'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                  'precipitation', 'rain']
    COLS_G_MAX = ['sunshine_duration', 'month', 'day', 'hour','cluster']
    COLS_G_INT = ['cluster', 'accident_next', 'accidentados_next', 'accidentes_next', 'gravedad_next', 'is_day',
                  'hay_accidente']
    COLS_DROP = ['vmed']

    COLS_CYCLIC = [ 'month', 'day', 'hour','Dia_semana']

    dic = 'medidas'
    year0 = 2023
    lim_accidentados = 5
    Sscaler = StandardScaler()
    Mscaler = MinMaxScaler()


    for nodo, dict_ in G.nodes.items():

        for col in dict_[dic].columns:


            if col in COLS_G_STD:
                G.nodes[nodo][dic][col] = Sscaler.fit_transform(dict_[dic][[col]])

            elif col in COLS_G_MAX:
                G.nodes[nodo][dic][col] = Mscaler.fit_transform(dict_[dic][[col]])

            elif col in COLS_G_INT:
                G.nodes[nodo][dic][col] = dict_[dic][col].astype(int)


            if col == 'year':
                G.nodes[nodo][dic][col] = G.nodes[nodo][dic][col] - year0

            elif col in COLS_DROP:
                G.nodes[nodo][dic].drop(columns=col)

            if col in COLS_CYCLIC:
                G.nodes[nodo][dic]=cyclic_encoder(G.nodes[nodo][dic],col)

        dict_[dic].loc[dict_[dic]['accidentados_next']>lim_accidentados,'accidentados_next'] = lim_accidentados
        binned = pd.cut(dict_[dic]['accident_next'], bins = bins_acc, labels = num_acc)
        dict_[dic].loc[:, 'accident_next'] = binned.astype(np.int32)

        dict_[dic] = balance_tiempos_cat(dict_[dic],num_acc)

    return G

def balance_tiempos_cat(df,num):
    N0=(df['hay_accidente']==True).sum()

    for nbin in num[1:-1]:# quitamos el 0 y el 3 que ya están considerados
        # Filtrar los índices correspondientes a la categoría actual

        index_cat = df.loc[df['accident_next'] == nbin].index
        # Seleccionar aleatoriamente `num_ceros` índices, o menos si no hay suficientes
        num_sel = min(len(index_cat), N0)
        filtro_cat = np.random.choice(index_cat, num_sel, replace = False)

        # Asignar 1s en la máscara en las posiciones seleccionadas

        df.loc[filtro_cat,'pad'] = True

    return df






# def sincronizar_indices_duplicados(G):
#     max_repeticiones = pd.Series(dtype = int)
#
#     for n, data in G.nodes.items():
#         df = data['medidas']
#
#         # Contar cuántas veces aparece cada índice (timestamp)
#         repeticiones = df['fecha'].value_counts()
#
#         # Actualizar el máximo número de repeticiones para cada timestamp
#         max_repeticiones = max_repeticiones.combine(repeticiones, max, fill_value = 0)
#
#     # Paso 2: Asegurarnos de que cada DataFrame tenga los mismos índices duplicados
#     df_list_sincronizados = []
#
#     for n, data in G.nodes.items():
#
#         df = data['medidas']
#
#         # Crear una lista para almacenar las filas replicadas
#         filas_replicadas = []
#
#         for f, count in max_repeticiones.items():
#             # Seleccionar todas las filas que coinciden con el timestamp actual
#             filas = df[df['fecha'] == f]
#
#             # Si hay menos filas de las necesarias, replicar filas hasta alcanzar 'count'
#             if len(filas) < count:
#                 # Replicar filas para que el DataFrame tenga el mismo número que 'max_repeticiones'
#                 filas_replicadas.append(filas.sample(count, replace = True))
#             else:
#                 # Si ya hay suficientes filas, no replicar
#                 filas_replicadas.append(filas)
#
#         # Concatenar todas las filas replicadas
#         df_replicado = pd.concat(filas_replicadas).sort_values(by = 'fecha').reset_index(drop = True)
#
#         G.nodes[n]['medidas'] = df_replicado
#
#     return G


# Definir el número de pasos previos N
# N = 10
#
# # Generar secuencias para un nodo del grafo
# df_node = G.nodes[0]['medidas']
# X_sequences, y_labels = create_sequences(df_node, N)
#
# print("Dimensión de las etiquetas (y):", y_labels.shape)  # (num_timesteps - N, num_labels)
# print("Dimensión de las secuencias de entrada (X):", X_sequences.shape)  # (num_timesteps - N, N+1, num_features)


