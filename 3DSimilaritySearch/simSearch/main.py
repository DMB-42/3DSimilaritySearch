from data import prepareInputObject, loadData
from loadModel import loadModel, createFeatureExtractor
from similaritySearch import calculateDistances, obtainMostSimilarObjectIndex
from visualize import visualizeResults
import numpy as np

def main():
    
    #Definimos el número de puntos de cada nube, el número total de clases y el tamaño de lote
    NUM_POINTS = 2048
    NUM_CLASSES = 10
    BATCH_SIZE = 32

    #Cargamos los datos desde los archivos guardados durante el proceso de entrenamiento del modelo
    trainData, trainLabels, testData, testLabels = loadData('modelTrain/preprocessedDS/train_points.npy', 
                                                            'modelTrain/preprocessedDS/train_labels.npy', 
                                                            'modelTrain/preprocessedDS/test_points50.npy', 
                                                            'modelTrain/preprocessedDS/test_labels50.npy')

    #Cargamos el modelo entrenado previamente
    model = loadModel('modelTrain/models/pointnetE50V5678.h5')

    #Creamos el extractor de características a partir del modelo entrenado previamente
    featureExtractor = createFeatureExtractor(model)

    #Extraemos las características de todas las nubes de puntos cargadas previamente
    allFeatures = featureExtractor.predict(trainData, BATCH_SIZE)

    #Inicializamos la matriz de confusión
    confusionMatrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    #Creamos una variable para guardar las nubes de puntos para su posterior visualización
    objectsVisualize = []

    #Recorremos uno a uno todos los objetos de testeo
    for i in range(testData.shape[0]):
        
        #Preparamos el objeto de entrada
        inputObjectTf = prepareInputObject(testData[i], NUM_POINTS)
        
        #Extraemos las características del objeto de entrada
        inputFeatures = featureExtractor.predict(inputObjectTf)
        
        #Calculamos las distancias euclídeas entre el objeto de entrada y todas las nubes de puntos
        distances = calculateDistances(inputFeatures, allFeatures)
        
        #Obtenemos el índice de la nube de puntos más similar
        mostSimilarIndex = obtainMostSimilarObjectIndex(distances)
        
        #Obtenemos la etiqueta verdadera
        trueLabel = testLabels[i]
        
        #Incrementamos el recuento correspondiente en la matriz de confusión
        predictedLabel = trainLabels[mostSimilarIndex]
        print("Etiqueta Real: " + str(trueLabel) + ". Etiqueta Predicha: " + str(predictedLabel))
        confusionMatrix[trueLabel, predictedLabel] += 1
        
        if i % 49 == 0 and i != 0:
            #Preparamos el objeto más similar predicho
            mostSimilarObject = prepareInputObject(trainData[mostSimilarIndex], NUM_POINTS)

            #Guardamos el objeto de entrada y su etiqueta y el objeto más similar predicho y su etiqueta
            objectsVisualize.append((inputObjectTf, trueLabel, mostSimilarObject, predictedLabel))
            
    #Visualizamos algunos objetos de entrada y sus predicciones más similares
    visualizeResults(objectsVisualize)

    #Imprimimos la matriz de confusión
    print("Matriz de Confusión:")
    print(confusionMatrix)
    
    #Calculamos la traza
    trace = np.trace(confusionMatrix)
    
    #Calculamos el porcentaje de acierto de nuestra búsqueda por similitud
    acierto = trace / testData.shape[0]
    print("El porcentaje de acierto es: %.2f%%" % (acierto * 100))

if __name__ == "__main__":
    main()