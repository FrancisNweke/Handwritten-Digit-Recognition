from tensorflow.keras.models import Model, load_model


# To learn more, please visit this website: https://machinelearningmastery.com/save-load-keras-deep-learning-models/

def save_model(model: Model, filename):
    model.save(f'data/{filename}.h5')

    return f'Model saved as {filename}.h5'


# Load a saved model from your device.
def load_model(filename):
    model: Model = load_model(f'data/{filename}.h5')

    return model
