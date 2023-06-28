import numpy as np
import tensorflow as tf
import trimesh

def loadData(trainPoints, testPoints):
    trPoints = np.load(trainPoints)
    tePoints = np.load(testPoints)
    allData = np.concatenate([trPoints, tePoints], axis=0)
    return allData

def prepareInputObject(inputObjectPath):
    inputObject = trimesh.load(inputObjectPath).sample(2048)
    inputObject = inputObject.reshape(1, 2048, 3)
    inputObjectTf = tf.convert_to_tensor(inputObject, dtype=tf.float64)
    return inputObjectTf