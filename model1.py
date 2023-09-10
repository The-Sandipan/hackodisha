from tensorflow import keras

def load_model():
    model1 = keras.models.load_model('tumor_classification_model2.h5')
    return model1
