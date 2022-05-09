from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from netmodels.multilayer import multilayer_model, process_multilayer_data
from netmodels.convnetlayer import convNet_model, process_cnn_data
from netmodels.complexcnnlayer import complex_cnn_model
from netmodels.save_load_net import save_model, load_model
import time

"""
The data is downloaded from MNIST if not available already. The file is kept in this directory: 
/.keras/datasets/mnist.pkl.gz (size is 15MB)

Note: channels can be called the colors. for RGB, it has a value of 3. While gray scale is 1.
"""
# Load data from Modified National Institute of Standards and Technology
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('\t\t\t\t\t\tHandwritten Digit Recognition using MNIST')

user_choice = int(input(
    'Please select the algorithm you wish to use. \nPress 1 for Multilayer Perceptron\nPress 2 for Simple '
    'Convolutional Neural Network\nPress 3 for Complex Convolutional Neural Network\n\nEnter 1, 2 or 3 here: '))

valid_input = False

if user_choice == 1:  # Data preprocessing for multilayer perceptron
    valid_input = True
    num_pixels, train_X, test_X = process_multilayer_data(train_X, test_X)
elif user_choice == 2:  # Data preprocessing for a simple CNN
    valid_input = True
    train_X, test_X = process_cnn_data(train_X, test_X)
elif user_choice == 3:  # Data preprocessing for a complex CNN
    valid_input = True
    train_X, test_X = process_cnn_data(train_X, test_X)
else:
    print('Input is wrong and/or out-of-range.')

if valid_input:
    # Normalize inputs from 0-255 to 0-1
    train_X = train_X / 255
    test_X = test_X / 255

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    num_classes = test_y.shape[1]

    # Start time
    start_time = time.perf_counter()

    if user_choice == 1:  # Multilayer perceptron neural network - ANN
        model = multilayer_model(num_pixels, num_classes)
    elif user_choice == 2:  # Simple CNN
        model = convNet_model(num_classes)
    elif user_choice == 3:  # Complex CNN
        model = complex_cnn_model(num_classes)

    model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=10, batch_size=600, verbose=2)

    loss_function, accuracy = model.evaluate(test_X, test_y, verbose=0)

    # Stop time
    stop_time = time.perf_counter()

    # Calculate the difference btw stop time and start time
    process_time = stop_time - start_time

    print("\nError: {:.2f}".format(round(loss_function, 2)))
    print("Accuracy: {:.2f}%".format(round(accuracy * 100, 2)))
    print(f'The process time: {process_time} secs')

    # Press the green button in the gutter to run the script.
    # if __name__ == '__main__':
    # print_hi('PyCharm')

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
else:
    print('Please exit program.....')