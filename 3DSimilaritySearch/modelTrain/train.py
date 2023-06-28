def train_model(model, train_dataset, test_dataset, epochs, model_path):
    
    #Entrenamos el modelo con el conjunto de entrenamiento, 
    #especificamos el número de épocas y el conjunto de validación
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
    
    #Guardamos el modelo entrenado en la ubicación especificada
    model.save(model_path)
