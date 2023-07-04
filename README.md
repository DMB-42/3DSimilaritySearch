# 3DSimilaritySearch
Trabajo de Fin de Grado basado en la búsqueda de datos 3D mediante técnicas de Deep Learning, en este caso, usando una red PointNet para entrenar un modelo. A partir de dicho modelo, creamos un extractor de características para comparar nubes de datos 3D usando la distancia euclídea.

Tecnologías Utilizadas

A lo largo del desarrollo de este proyecto, se utilizaron diversas tecnologías para implementar y desplegar la solución propuesta. 
A continuación, se detallan las tecnologías utilizadas y las razones de su elección.

• Sistema Operativo: El desarrollo y la ejecución del proyecto se llevaron a cabo en
Windows 10. Este sistema operativo fue elegido por su facilidad de uso, su compatibilidad con una variedad de software y hardware, y la familiaridad del desarrollador con
el entorno.

• Unidad de Procesamiento Gráfico (GPU): Para el procesamiento de los cálculos
intensivos requeridos en el entrenamiento de modelos de Deep Learning, se utilizó una
GPU NVIDIA GeForce GTX 1050. Esta GPU es compatible con CUDA y cuDNN, y
permitió una ejecución mucho más rápida del entrenamiento de modelos en comparación
con el uso de una CPU solamente. La elección de una GPU NVIDIA se debe a la
excelente integración que estas tarjetas tiene con las bibliotecas de TensorFlow, CUDA
y cuDNN, permitiendo una aceleración considerable en las tareas de entrenamiento y
predicción del modelo.

• CUDA (v. 11.2) y cuDNN (v. 8.1.0): Estas dos tecnologías de NVIDIA se utilizaron
para habilitar la aceleración de GPU para el entrenamiento de la red neuronal en
TensorFlow. CUDA es una plataforma de computación paralela que permite aumentar
drásticamente la velocidad de computación usando la potencia de las GPUs, y cuDNN
es una biblioteca de primitivas para las redes neuronales profundas que proporciona
rutinas de alto rendimiento para las operaciones estándar en las DNNs.

• Entorno de Desarrollo Integrado (IDE): Se utilizó Visual Studio Code (VS Code) como el entorno de desarrollo principal para este proyecto. VS Code es un editor
de código fuente ligero pero potente que puede ser extendido con complementos para
soportar una variedad de lenguajes y frameworks de programación. VS Code proporciona una interfaz de usuario amigable y funcionalidades como resaltado de sintaxis,
autocompletado inteligente, y soporte de depuración, lo que hace que el proceso de
desarrollo sea más eficiente.

• Lenguaje de programación: El lenguaje de programación principal utilizado en este
proyecto fue Python (versión 3.10.11). Python es ampliamente utilizado en la comunidad
de ciencia de datos y machine learning debido a su sintaxis limpia y fácil de leer, así
como su amplio ecosistema de bibliotecas de ciencia de datos.

• Bibliotecas de Python:

– TensorFlow (v. 2.10.1): TensorFlow es una biblioteca de código abierto para
la computación numérica y el aprendizaje automático a gran escala. TensorFlow
proporciona una serie de características que permiten el desarrollo fácil y eficiente
de modelos de deep learning. En este proyecto, se utilizó TensorFlow para construir
y entrenar la arquitectura PointNet.

– Numpy (v. 1.24.3): Numpy es una biblioteca esencial que proporciona soporte
para matrices y matrices multidimensionales de gran tamaño. Viene equipada con
una amplia colección de funciones matemáticas que facilitan operaciones de alto
nivel sobre estas estructuras de datos. En este proyecto, uno de sus diversos usos
fue para guardar (y cargar) los datos ya procesados.

– Matplotlib (v. 3.7.1): Matplotlib es una biblioteca de visualización de datos que
permite la creación de gráficos y visualizaciones de alta calidad. Es una herramienta
fundamental para la exploración y presentación de datos, permitiendo una amplia
variedad de estilos y formatos de gráficos. En este proyecto, fue usada para mostrar
nubes de puntos 3D.

– Trimesh (v. 3.22.0): Trimesh es una biblioteca para cargar, guardar, visualizar y
manipular mallas triangulares, que se utilizó para manipular y procesar las nubes
de puntos 3D.
