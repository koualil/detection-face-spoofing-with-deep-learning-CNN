import tensorflow as tf

def load_model():
    return tf.keras.models.load_model('/Users/mkoualil/Desktop/backend/mobilenetv2-epoch_02.hdf5')

def predict(model, input_data):
    return model.predict(input_data)
