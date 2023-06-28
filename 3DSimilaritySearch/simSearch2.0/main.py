from data import loadData, prepareInputObject
from loadModel import loadModel, createFeatureExtractor
from similaritySearch import calculateDistances, obtainMostSimilarObjectsIndexes
from visualize import visualizeResults

def main():
    #Definimos el path del objeto que se utilizará como input
    inputObjectPath = "simSearch2.0/inputs/laptop_0003.off"

    #Definimos el número de objetos que se devolverán
    numObjectsToReturn = 5

    #Cargamos los datos desde los archivos guardados durante el proceso de entrenamiento del modelo
    allData = loadData('modelTrain/preprocessedDS/train_points.npy', 'modelTrain/preprocessedDS/test_points.npy')

    #Cargamos el modelo entrenado previamente
    model = loadModel('modelTrain/models/pointnetE50V5678.h5')

    #Creamos el extractor de características a partir del modelo entrenado previamente
    featureExtractor = createFeatureExtractor(model)

    #Extraemos las características de todas las nubes de puntos cargadas previamente
    allFeatures = featureExtractor.predict(allData, batch_size=32)

    #Preparamos el objeto de entrada
    inputObjectTf = prepareInputObject(inputObjectPath)

    #Extraemos las características del objeto de entrada
    inputFeatures = featureExtractor.predict(inputObjectTf)

    #Calculamos las distancias euclídeas entre el objeto de entrada y todas las nubes de puntos
    distances = calculateDistances(inputFeatures, allFeatures)

    #Obtenemos los índices de las nubes de puntos más similares
    mostSimilarIndices = obtainMostSimilarObjectsIndexes(distances, numObjectsToReturn)

    #Obtenemos las nubes de putnos más similares
    mostSimilarObjects = allData[mostSimilarIndices]

    #Visualizamos el objeto de entrada junto con las nubes de puntos más similares
    visualizeResults(inputObjectTf.numpy(), mostSimilarObjects, numObjectsToReturn)

if __name__ == "__main__":
    main()