import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

#Fijamos la semilla del generador de números aleatorios para garantizar la reproducibilidad de los experimentos
tf.random.set_seed(1807)

#Cada capa de convolución (convolution) y totalmente conectada (fully-connected),con excepción de las capas finales,
#consta de Convolution / Dense -> Batch Normalization -> ReLU Activation
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

#Regularizador personalizado para garantizar que la matriz de transformación sea ortogonal
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
        
    #Funciones necesarias para el serializado/guardado del modelo  
    def get_config(self):
        return {"num_features": self.num_features, "l2reg": self.l2reg}
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

#Red T-Net que devuelve la matriz de transformación
def tnet(inputs, num_features):
    
    #Inicializamos el bias como la matriz de identidad
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    
    #Aplicamos una transformación afín a las características de entrada
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

#Construimos la red completa PointNet (los pesos son la mitad de la red PointNet original)
def build_model(num_points, num_classes):
    inputs = keras.Input(shape=(num_points, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    return model