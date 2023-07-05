# 3DSimilaritySearch
Trabajo de Fin de Grado basado en la búsqueda de datos 3D mediante técnicas de Deep Learning, en este caso, usando una red PointNet para entrenar un modelo. A partir de dicho modelo, creamos un extractor de características para comparar nubes de datos 3D usando la distancia euclídea.

## Tecnologías Utilizadas

A lo largo del desarrollo de este proyecto, se utilizaron diversas tecnologías para implementar y desplegar la solución propuesta. 
A continuación, se detallan las tecnologías utilizadas:

• **Sistema Operativo**: El desarrollo y la ejecución del proyecto se llevaron a cabo en
Windows 10.

• **Unidad de Procesamiento Gráfico (GPU)**: Para el procesamiento de los cálculos
intensivos requeridos en el entrenamiento de modelos de Deep Learning, se utilizó una
GPU NVIDIA GeForce GTX 1050.

• **CUDA (v. 11.2) y cuDNN (v. 8.1.0)**

• **Entorno de Desarrollo Integrado (IDE)**: Se utilizó Visual Studio Code (VS Code) como el entorno de desarrollo principal para este proyecto.

• **Lenguaje de programación**: El lenguaje de programación principal utilizado en este
proyecto fue Python (versión 3.10.11).

• **Bibliotecas de Python**:

  – **TensorFlow (v. 2.10.1)**

  – **Numpy (v. 1.24.3)**

  – **Matplotlib (v. 3.7.1)**

  – **Trimesh (v. 3.22.0)**

## Guía de utilización

1. En primer lugar, se recomienda preprocesar el dataset ModelNet10 (para almacenar los datos preprocesados y el mapeo de clases en la carpeta 'modelTrain/preprocessedDS'). Como se puede observar, la carpeta 'preprocessedDS' ya contiene dos archivos: test_labels50.npy y test_points50.npy. Estos archivos contienen 50 nubes de puntos de cada clase (test_points50.npy) perteneciente a ModelNet10 y su correspondiente clase (test_labels50.npy). Estos archivos son fundamentales para obtener una matriz de confusión que de unos resultados fiables, ya que necesita la misma cantidad de entradas de cada clase.

2. Si no se quiere utilizar un modelo ya entrenado (se pueden encontrar en 'modelTrain/models'), se recomienda ejecutar el script 'modelTrain/main.py' indicando el número de épocas y la semilla para generar número aleatorios (se puede modificar en el archivo 'modelTrain/model.py'). La notación que se ha utilizado para nombrar a los modelos ha sido: 'pointnetExxVyyyy.h5', siendo 'Exx' el número de épocas (por ejemplo, 'E25' para 25 épocas) y 'Vyyyy' la semilla aleatoria para TensorFlow (por ejemplo, 'V1807' para la semilla 1807).
   
3. Para ejecutar el módulo 'simSearch' (búsqueda por similitud), ejecutar el script 'simSearch/main.py'. Se pueden usar otros datos de entrenamiento y prueba, simplemente poner los archivos '.npy' en los que se hayan procesado los datos deseados. Al igual que se puede usar otro modelo para crear el extractor de características. Simplemente se cambiaría la ruta a la del modelo deseado.

4. Para el módulo experimental 'simSearch2.0', se pueden añadir otros objetos de entrada. Simplemente se han de añadir a la carpeta 'simSearch2.0/inputs'. Estos objetos de entrada han de estar en formato '.off', principalmente son modelos del dataset ModelNet40 pertenecientes a clases que no conforman ModelNet10.
