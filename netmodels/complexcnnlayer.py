from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model


# A complex CNN with convolution layers, pooling layer (Maximum), dropout, flatten and fully connected layers.
def complex_cnn_model(num_classes):
    visible_layer = Input(shape=(28, 28, 1))
    conv_layer_1 = Conv2D(20, kernel_size=(5, 5), activation='relu')(visible_layer)
    pooling_layer_1 = MaxPooling2D(pool_size=(2, 2))(conv_layer_1)
    third_layer = Conv2D(15, kernel_size=(4, 4), activation='relu')(pooling_layer_1)
    pooling_layer_2 = MaxPooling2D(pool_size=(2, 2))(third_layer)
    conv_layer_2 = Conv2D(7, kernel_size=(3, 3), activation='relu')(pooling_layer_2)
    dropout_layer = Dropout(rate=0.2)(conv_layer_2)
    transform_matrix_to_vector_layer = Flatten()(dropout_layer)
    fc_layer_1 = Dense(128, activation='relu')(transform_matrix_to_vector_layer)
    fc_layer_2 = Dense(50, activation='relu')(fc_layer_1)
    output_layer = Dense(num_classes, activation='softmax')(fc_layer_2)

    model = Model(inputs=visible_layer, outputs=output_layer)

    plot_model(model, to_file='data/complexCovNet.png', show_shapes=True)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
