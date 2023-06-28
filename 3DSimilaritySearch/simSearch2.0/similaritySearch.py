import numpy as np
from scipy.spatial import distance_matrix

def calculateDistances(inputFeatures, allFeatures):
    inputFeaturesReshaped = inputFeatures.reshape((inputFeatures.shape[0], -1))
    allFeaturesReshaped = allFeatures.reshape((allFeatures.shape[0], -1))

    distances = distance_matrix(inputFeaturesReshaped, allFeaturesReshaped)
    return distances

def obtainMostSimilarObjectsIndexes(distances, numObjectsToReturn):
    mostSimilarIndices = np.argsort(distances[0])[:numObjectsToReturn]
    return mostSimilarIndices