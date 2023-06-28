import pickle
from matplotlib import pyplot as plt
    
def visualizeResults(results):
    
    #Cargamos el mapeo de clases desde el archivo guardado anteriormente
    with open('modelTrain/preprocessedDS/class_map.pkl', 'rb') as f:
        CLASS_MAP = pickle.load(f)

    #Obtenemos el número total de objetos a visualizar
    num_results = len(results)

    #Definimos el layout de los subplots, que será de 2 filas y 6 columnas
    rows = 2
    cols = 6

    #Dividimos la lista de objetos en dos mitades para crear dos figuras
    first_half = results[:num_results//2]
    second_half = results[num_results//2:]

    #Creamos una figura para la primera mitad de objetos
    fig1 = plt.figure(figsize=(cols * 10, rows * 10))
    
    #Iteramos sobre cada objeto de la primera mitad.
    for i, (inputObject, inputLabel, similarObject, similarLabel) in enumerate(first_half):
        
        #Ponemos el objeto de entrada en un subplot junto con un texto indicando a que clase pertenece
        ax = fig1.add_subplot(rows, cols, i*2+1, projection='3d')
        ax.scatter(inputObject[0, :, 0], inputObject[0, :, 1], inputObject[0, :, 2])
        ax.set_title('{}. Input object - Class {}'.format(inputLabel, CLASS_MAP[inputLabel]), fontsize=9)
        ax.dist = 8

        #Ponemos el objeto más similar en otro subplot junto con un texto indicando a que clase pertenece
        ax = fig1.add_subplot(rows, cols, i*2+2, projection='3d')
        ax.scatter(similarObject[0, :, 0], similarObject[0, :, 1], similarObject[0, :, 2])
        ax.set_title('{}. Similar object - Class {}'.format(inputLabel, CLASS_MAP[similarLabel]), fontsize=9)
        ax.dist = 8

    #Ajustamos el espacio entre los subplots y mostramos la primera figura
    plt.subplots_adjust(wspace=1, hspace=0.2)
    plt.show()

    # Creamos otra figura para la segunda mitad de los objetos
    fig2 = plt.figure(figsize=(cols * 10, rows * 10))
    
    #Iteramos sobre cada objeto de la segunda mitad
    for i, (inputObject, inputLabel, similarObject, similarLabel) in enumerate(second_half):
        
        #Ponemos el objeto de entrada en un subplot junto con un texto indicando a que clase pertenece
        ax = fig2.add_subplot(rows, cols, i*2+1, projection='3d')
        ax.scatter(inputObject[0, :, 0], inputObject[0, :, 1], inputObject[0, :, 2])
        ax.set_title('{}. Input object - Class {}'.format(inputLabel, CLASS_MAP[inputLabel]), fontsize=8)
        ax.dist = 8

        #Ponemos el objeto más similar en otro subplot junto con un texto indicando a que clase pertenece
        ax = fig2.add_subplot(rows, cols, i*2+2, projection='3d')
        ax.scatter(similarObject[0, :, 0], similarObject[0, :, 1], similarObject[0, :, 2])
        ax.set_title('{}. Similar object - Class {}'.format(inputLabel, CLASS_MAP[similarLabel]), fontsize=8)
        ax.dist = 8

    #Ajustamos el espacio entre los subplots y mostramos la segunda figura
    plt.subplots_adjust(wspace=1, hspace=0.2)
    plt.show()
