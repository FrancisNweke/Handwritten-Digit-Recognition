from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

"""
We are going to have 3 hidden layers in our network.
The first layer has n number of  nodes with the activation function as ReLU
The second layer has half of n number of nodes with the activation function as ReLU
The output layer has x number of nodes with the activation function as Softmax

Network architecture: pixel_number/pixel_number/half of pixel_number/num_classes

Note: the visible layer is also known as the input layer
"""


# Flatten - convert the matrix of pixel values (28 by 28) to 784 vector space. - Multilayer
def process_multilayer_data(train_X, test_X):
    pixel_number = train_X.shape[1] * train_X.shape[2]
    train_X = train_X.reshape(train_X.shape[0], pixel_number).astype('float32')
    test_X = test_X.reshape(test_X.shape[0], pixel_number).astype('float32')

    return pixel_number, train_X, test_X


# Define the multilayer perceptron network - ANN
def multilayer_model(pixel_number, num_classes):
    visible_layer = Input(shape=(pixel_number,))
    first_layer = Dense(pixel_number, activation='relu', kernel_initializer='normal')(visible_layer)
    second_layer = Dense(pixel_number / 2, activation='relu', kernel_initializer='normal')(first_layer)
    multiclass_output = Dense(num_classes, activation='softmax', kernel_initializer='normal')(second_layer)

    model = Model(inputs=visible_layer, outputs=multiclass_output)

    plot_model(model, to_file='data/multilayer.png', show_shapes=True)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

    # predictions = model.predict(test_X, test_y, verbose=2)

    # print(predictions.any())
