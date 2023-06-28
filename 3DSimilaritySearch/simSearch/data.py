import numpy as np
import tensorflow as tf

def loadData(trainPointsPath, trainLabelsPath, testPointsPath, testLabelsPath):
    
    #Cargamos las nubes de puntos de entrenamiento desde un archivo numpy
    trainPoints = np.load(trainPointsPath)
    
    #Cargamos las etiquetas de entrenamiento desde un archivo numpy
    trainLabels = np.load(trainLabelsPath)
    
    #Cargamos las nubes de puntos de testeo desde un archivo numpy
    testPoints = np.load(testPointsPath)
    
    #Cargamos las etiquetas de testeo desde un archivo numpy
    testLabels = np.load(testLabelsPath)

    #Devolvemos las nubes de puntos y las etiquetas de los conjuntos de entrenamiento y testeo
    return trainPoints, trainLabels, testPoints, testLabels

def prepareInputObject(inputObject, numPoints):
    
    #Reformateamos el objeto de entrada
    inputObject = inputObject.reshape(1, numPoints, 3)
    
    #Convertimos el objeto de entrada en un tensor TensorFlow
    inputObjectTf = tf.convert_to_tensor(inputObject, dtype=tf.float64)

    #Devolvemos el objeto de entrada reformateado
    return inputObjectTf