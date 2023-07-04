import os
import glob
import trimesh
import pickle
import numpy as np
import tensorflow as tf

#Definimos el directorio de datos
DATA_DIR = "modelTrain/modelnet/ModelNet10"

#Función para procesar el dataset
def parse_dataset(num_points):

    #Inicializamos listas para almacenar los puntos y etiquetas de entrenamiento y testeo
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    
    #Inicializamos un diccionario para mapear las clases a índices
    class_map = {}
    
    #Obtenemos los nombres de las carpetas en el directorio de datos
    folders = glob.glob(os.path.join(DATA_DIR, "*"))
    folders = [folder for folder in folders if "README" not in folder]

    #Procesamos cada carpeta
    for i, folder in enumerate(folders):
        print("Procsando clase: {}".format(os.path.basename(folder)))
        
        #Guardamos el nombre de la carpeta con un ID para recuperarlo más tarde
        class_map[i] = folder.split(os.sep)[-1]
        
        #Recopilamos todos los archivos
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        #Cargamos y procesamos cada archivo
        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    #Devolvemos los puntos y etiquetas de entrenamiento y testeo, y el mapa de clases
    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )
    
#Función de aumento para agitar y mezclar el conjunto de datos de entrenamiento.
def augment(points, label):
    
        #Agitamos los puntos
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        
        #Mezclamos los puntos
        points = tf.random.shuffle(points)
        return points, label

#Función para preparar los conjuntos de datos
def prepare_datasets(num_points, batch_size):

    #Procesamos los datos
    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        num_points
    )
    
    #Guardamos los datos procesados
    np.save('modelTrain/preprocessedDS/train_points.npy', train_points)
    np.save('modelTrain/preprocessedDS/test_points.npy', test_points)
    np.save('modelTrain/preprocessedDS/train_labels.npy', train_labels)
    np.save('modelTrain/preprocessedDS/test_labels.npy', test_labels)
    with open('modelTrain/preprocessedDS/class_map.pkl', 'wb') as f:
        pickle.dump(CLASS_MAP, f) 

    #Creamos los conjuntos de datos de entrenamiento y prueba
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    #Aplicamos el aumento de datos y agrupamos en lotes
    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)

    #Devolvemos los conjuntos de datos y el mapa de clases
    return train_dataset, test_dataset, CLASS_MAP

# Función para procesar los conjuntos de datos a partir de cargar los datos
def load_data(batch_size):
    
    #Cargamos los datos procesados previamente
    train_points = np.load('modelTrain/preprocessedDS/train_points.npy')
    test_points = np.load('modelTrain/preprocessedDS/test_points.npy')
    train_labels = np.load('modelTrain/preprocessedDS/train_labels.npy')
    test_labels = np.load('modelTrain/preprocessedDS/test_labels.npy')
    with open('modelTrain/preprocessedDS/class_map.pkl', 'rb') as f:
        CLASS_MAP = pickle.load(f)

    #Creamos los conjuntos de datos de entrenamiento y prueba
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    #Aplicamos el aumento de datos y agrupamos en lotes
    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)

    #Devolvemos los conjuntos de datos y el mapa de clases
    return train_dataset, test_dataset, CLASS_MAP