import tensorflow as tf
import os
import sys

#Obtenemos el directorio actual
current_directory = os.path.dirname(os.path.abspath(__file__))

#Navegamos un nivel arriba para llegar a la carpeta padre
parent_directory = os.path.dirname(current_directory)

#Añadimos la carpeta padre a sys.path
sys.path.insert(0, parent_directory)

from tensorflow import keras
from modelTrain.model import OrthogonalRegularizer

def loadModel(modelPath):
    custom_objects = {"OrthogonalRegularizer": OrthogonalRegularizer}
    model = tf.keras.models.load_model(modelPath, custom_objects=custom_objects)
    return model

def createFeatureExtractor(model):
    featureExtractor = keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
    return featureExtractor