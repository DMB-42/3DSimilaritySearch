import numpy as np
from scipy.spatial import distance_matrix

def calculateDistances(inputFeatures, allFeatures):
    
    #Redimensionamos las características a 2D para poder usar la matriz de distancias
    inputFeaturesReshaped = inputFeatures.reshape((inputFeatures.shape[0], -1))
    allFeaturesReshaped = allFeatures.reshape((allFeatures.shape[0], -1))

    #Calculamos la distancia euclídea (mediante la matriz de distancias) entre las características
    #de entrada y todas las características
    distances = distance_matrix(inputFeaturesReshaped, allFeaturesReshaped)
    
    #Devolvemos las distancias euclídeas
    return distances

def obtainMostSimilarObjectIndex(distances):
    
    #Ordenamos los índices de la matriz de distancias en función de las distancias más cortas a más largas
    mostSimilarIndexes = np.argsort(distances[0])
    
    #Obtenemos el índice del objeto más similar (la menor distancia)
    mostSimilarIndex = mostSimilarIndexes[0]
    
    #Devolvemos el índice
    return mostSimilarIndex