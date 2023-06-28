import tensorflow as tf

#Comentamos una función de preprocessing en función de si queremos preprocesar los datos o ya lo hicimos y queremos cargarlos
#from preprocessing import prepare_datasets
from preprocessing import load_data

from model import build_model
from train import train_model
from visualize import visualize_predictions

#Habilitamos el crecimiento de memoria en las GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#Definimos el número de puntos de cada nube, el número total de clases y el tamaño de lote
NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

#Preparamos los conjuntos de entrenamiento y de prueba
#train_dataset, test_dataset, CLASS_MAP = prepare_datasets(NUM_POINTS, BATCH_SIZE)

#Preparamos los conjuntos de entrenamiento y de prueba a partir de archivos ya procesados
train_dataset, test_dataset, CLASS_MAP = load_data(BATCH_SIZE)

#Creamos y compilamos el modelo
model = build_model(NUM_POINTS, NUM_CLASSES)

#Entrenamos y guardamos el modelo en un archivo
train_model(model, train_dataset, test_dataset, 25, 'modelTrain/models/pointnetE25V1807.h5')

#Mostramos algunas de las predicciones
visualize_predictions(model, test_dataset, CLASS_MAP)