import tensorflow as tf
import os
import sys
from tensorflow import keras

#Obtenemos el directorio actual
current_directory = os.path.dirname(os.path.abspath(__file__))

#Navegamos un nivel arriba para llegar a la carpeta pointnet
pointnet_directory = os.path.dirname(current_directory)

#Añadimos la carpeta pointnet a sys.path
sys.path.insert(0, pointnet_directory)

from modelTrain.model import OrthogonalRegularizer

def loadModel(modelPath):
    
    #Creamos un diccionario para mapear el nombre de la clase OrthogonalRegularizer
    custom_objects = {"OrthogonalRegularizer": OrthogonalRegularizer}
    
    #Cargamos el modelo entrenado previamente
    model = tf.keras.models.load_model(modelPath, custom_objects=custom_objects)
    
    #Devolvemos el modelo cargado
    return model

def createFeatureExtractor(model):
    
    #Creamos el exctractor de características con las mismas entradas del modelo y las salidas de la undécima capa
    featureExtractor = keras.Model(inputs=model.inputs, outputs=model.layers[10].output)
    
    #Devolvemos el extractor de características
    return featureExtractor