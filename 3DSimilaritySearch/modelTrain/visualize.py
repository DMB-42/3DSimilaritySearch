import os
import matplotlib.pyplot as plt
import tensorflow as tf

#Función para visualizar algunas de las predicciones
def visualize_predictions(model, test_dataset, class_map, num_samples=8):
    
    #Tomamos un lote de datos del conjunto de prueba
    data = test_dataset.take(1)

    #Extraemos los puntos y las etiquetas de los datos
    points, labels = list(data)[0]
    
    #Nos quedamos con una muestra de los puntos y las etiquetas
    points = points[:num_samples, ...]
    labels = labels[:num_samples, ...]

    #Pasamos los puntos a través del modelo para obtener las predicciones
    preds = model.predict(points)
    
    #Nos quedamos con el índice de la mayor probabilidad para obtener la clase predicha
    preds = tf.math.argmax(preds, -1)

    #Convertimos los puntos a numpy para poder manipularlos en matplotlib
    points = points.numpy()

    #Creamos una figura para visualizar los resultados
    fig = plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        
        #Establecemos el título del gráfico con la clase predicha y la verdadera
        ax.set_title(
            "pred: {:}, label: {:}".format(
                os.path.basename(class_map[preds[i].numpy()]),
                os.path.basename(class_map[labels.numpy()[i]])
            )
        )
        
        #Quitamos los ejes para una mejor visualización
        ax.set_axis_off()
        
    #Mostramos la figura    
    plt.show()