
BASE_drive = 'C:\\JoseMNR\\OTROS__\\TFM\\'
DS_folder_sens = 'Sensores'
DS_folder_ubi = 'Ubicaciones'
DS_folder_acc = 'Accidentes'

CAL_FILE = 'calendario.csv'
GFILE = 'G_precompleto.pkl'
MED_FILE = 'medidas.pkl'
MED_FILE_CLEAN = 'medidas_limpio.pkl'
DS_file = 'DATASET.pkl'

COLS_UBI = ['tipo_elem', 'distrito', 'longitud', 'latitud']
COLS_SENS = ['intensidad', 'ocupacion', 'carga', 'vmed']

ord_ = ['cluster', 'fecha']
drop_col = ['error', 'id']
cols_v = ['id', 'fecha', 'intensidad', 'ocupacion',
          'carga', 'vmed', 'error']
cols_to_encode = ['tipo_elem']


PAST_COLS = ['intensidad', 'ocupacion', 'carga', 'vmed',
             'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
             'precipitation', 'rain', 'is_day', 'sunshine_duration']

grupo_edad = {
    'De 0 a 17 años': ['Menor de 5 años', 'De 6 a 9 años', 'De 10 a 14 años', 'De 15 a 17 años'],
    'De 18 a 30 años': ['De 18 a 20 años', 'De 21 a 24 años', 'De 25 a 29 años'],
    'De 31 a 45 años': ['De 30 a 34 años', 'De 35 a 39 años', 'De 40 a 44 años'],
    'De 46 a 60 años': ['De 45 a 49 años', 'De 50 a 54 años', 'De 55 a 59 años'],
    'Más mayores de 60': ['De 60 a 64 años', 'De 65 a 69 años', 'De 70 a 74 años', 'Más de 74 años'],
    'Desconocido': ['Desconocido']
}

gravedad = {
    'Sin gravedad': [14],
    'Leve': [1, 2, 5, 6, 7],
    'Grave': [3],
    'Fallecido': [4]
}

DROP_COLS = ['pad', 'is_day', 'relative_humidity_2m', 'rain', 'fecha', 'vmed']
LABEL_COLS = ['accident_next', 'accidentados_next', 'gravedad_next']

PLOTS_FOLDER='MODEL\\graphic_reports'
MODEL_FILE = 'modelo_guardado.pth'
TIME_FILE = 'Tiempo.csv'
METRICS_FILE = 'Metricas.csv'
TEST_METRICS_FILE =  'Test_Metrics.csv'
TEST_DATA = 'Test_data.pt'
PREDS_LABELS_FILE = 'Salidas.csv'
OPTUNA_FILE = 'OPTUNA_params.pkl'


# num_epochs = 1
# comp = 2
num_clusters = 200
timewindow = 4
dist_threshold=40
periodos = [426,121]
categorias_acc = ['accidente','inminente', 'cercano', 'no_previsible']
num_acc = [0, 1, 2, 3]
bins_acc = [-1,15,45,120, 1e12]
lim_accidentados= 5

num_epochs = 50
batch_size = 1

hidden = 64
lr=0.0024826669390874963
weight_decay= 1.24899416096733e-05
dropouts=[0.3577487499911839, 0.4087557724079046]
# lr=0.001
# num_epochs_ = 5
# weight_decay = 0.00001
# dropouts = [0.25,0.35]




finalPCAdim=18
out = [len(num_acc), lim_accidentados + 1, len(gravedad)]
loss_weights_ = [0.8,0.05,0.15]

# OPTUNA_FLAG = True
# phases_optuna=True
# num_epochs_optuna = [10, 20]
# n_trials = [5, 10]
# prop_optuna=[0.1,0.5]

# OPTUNA_FLAG = True
# phases_optuna=True
OPTUNA_FLAG = True
phases_optuna=True

# num_epochs_optuna = [15, 30]
# n_trials = [8, 10]

num_epochs_optuna = [15, 15]
n_trials = [8, 3]

# num_epochs_optuna = [2, 30]
# n_trials = [1, 10]

# prop_optuna=[0.3,0.7]
prop_optuna=[0.3,0.3]

patience = 8
min_delta = 0.2
sch_factor=0.3