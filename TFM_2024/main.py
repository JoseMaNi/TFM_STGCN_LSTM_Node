#!/usr/bin/env python3


from funcs import *
from CONSTANTS import *

print(f'NOW: {datetime.datetime.now().strftime('%d / %B / %Y - %H : %M : %S')}')

def main():

    # Lencoder = LabelEncoder()

    os.chdir(BASE_drive)
    print(f'NOW: {datetime.datetime.now().strftime('%d / %B / %Y - %H : %M : %S')}')

    if DS_file not in os.listdir():

        medidas, df = medidas_sensores(base_folder=BASE_drive,
                                       folders=[DS_folder_ubi,DS_folder_sens],
                                       cols=[COLS_UBI,cols_v,cols_to_encode],
                                       files=MED_FILE)
        print((medidas['05-2024.csv'].vmed != 0).sum())

        Centers,G =clustering(df,num_clusters,COLS_UBI)
        os.chdir(BASE_drive)


        print('Se mapean los id por cluster y se sacan los tiempos con medida en cada nodo___')
        print((medidas['05-2024.csv'].vmed != 0).sum())

        medidas,fechas_en_nodo = limpieza_medidas(medidas, df,
                                                  cols=[COLS_SENS, drop_col, ord_],
                                                  folder=DS_folder_sens,
                                                  file = MED_FILE_CLEAN)

        print((medidas['05-2024.csv'].vmed != 0).sum())

        datemin = pd.to_datetime(medidas['10-2022.csv'].iloc[0].fecha)
        datemax = pd.to_datetime(medidas['05-2024.csv'].iloc[-1].fecha)
        range_dates = pd.date_range(start = datemin,
                                    end = datemax,
                                    freq = '15min')

        acc_cluster,filtro_total = ETL_acc(Centers,
                                          folder_acc=DS_folder_acc,
                                          dates=[range_dates[0],
                                                 range_dates[-1],
                                                 fechas_en_nodo],
                                          pasos=timewindow,
                                          nodos = G.nodes.keys(),
                                          gravedad = gravedad)

        range_dates = range_dates.intersection(filtro_total)
        print(range_dates)
        # BIEN

        if GFILE not in os.listdir(BASE_drive):

            # print('antes completar grafo', G.nodes[0]['medidas']['fecha'])
            # print('ELEMENTOS',(medidas['05-2024.csv'].vmed != 0).sum(),'MEDIA ',medidas['05-2024.csv'].vmed.mean())
            G = completar_grafo(G,
                                medidas,
                                Centers,
                                ranges=range_dates,
                                folder=BASE_drive,
                                file=CAL_FILE,
                                dates=[datemin,datemax])


            with open(GFILE,'wb') as f:

                pickle.dump([G,datemin,datemax],f)
            print('SE GUARDA GRAFO COMPLETO en ',os.getcwd())

        else:
            print('HAY GRAFO COMPLETO, SE CARGA')

            with open(BASE_drive + '\\' + GFILE,'rb') as f:
                G,datemin,datemax = pickle.load(f)
        equal_size_graph (G)
        G = transform_dataset(G,acc_cluster,aggregate = True)
        # print(G.nodes[0]['medidas'].loc[~G.nodes[0]['medidas'].hay_accidente.isnull()])
        # print(G.nodes[150]['medidas'].vmed.mean())

        G = scale_encode(G,num_acc=num_acc,bins_acc = bins_acc)
        print('FINAL CHECK')
        equal_size_graph(G)

        with open(DS_file,'wb') as file:
            pickle.dump([G, acc_cluster],file)

        print(f'DATASET GUARDADO en {os.getcwd()}')
        print('___+++___')
        return G

    else:

        print('YA HAY UN DATASET')


        with open(DS_file,'rb') as f:
            G, acc_cluster = pickle.load(f)
        equal_size_graph(G)
        print("SE CARGA", DS_file)
        print('___+++___')
        return G


if __name__ == '__main__':
    print('+++___+++')
    G_, acc_cluster_, LOCAL=main()
    for var_name, var_value in LOCAL.items():
        globals()[var_name] = var_value  # Asignar cada variable al espacio de nombres global

    print('_______________________________________________________________')
