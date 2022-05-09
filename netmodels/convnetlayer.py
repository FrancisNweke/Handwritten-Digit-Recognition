from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model


# Reshape data to [samples/pixels][width][height][channels] - CNN
def process_cnn_data(train_X, test_X):
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1).astype('float32')
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1).astype('float32')

    return train_X, test_X


"""
Build a simple CNN.

Network architecture: 28*28/32:5*5/2*2/
"""


def convNet_model(num_classes):
    visible_layer = Input(shape=(28, 28, 1))
    first_layer = Conv2D(32, kernel_size=(5, 5), activation='relu')(visible_layer)
    pooling_layer = MaxPooling2D(pool_size=(2, 2))(first_layer)
    dropout_layer = Dropout(0.2)(pooling_layer)
    transform_matrix_to_vector_layer = Flatten()(dropout_layer)
    fourth_layer = Dense(128, activation='relu')(transform_matrix_to_vector_layer)
    output_layer = Dense(num_classes, activation='softmax')(fourth_layer)

    model = Model(inputs=visible_layer, outputs=output_layer)

    plot_model(model, to_file='data/convNet.png', show_shapes=True)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
