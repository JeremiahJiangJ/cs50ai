import cv2
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CHANNELS = 3
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    
    # model_performance = evaluate_models(images, labels)
    
    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
    


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    print(f"Loading Image Data From {data_dir}\n")

    images = []
    labels = []

    category_count = 0
    for folder in os.listdir(data_dir):
        folder_dir = os.path.join(data_dir, folder)
        category = int(folder)
        category_count += 1
        print(f"Category {category_count}/{NUM_CATEGORIES}")
        for file in os.listdir(folder_dir):
            file_dir = os.path.join(folder_dir, file)
            img = cv2.imread(file_dir)
            if img is None:
                print(f"img is None, check file path = {file_dir}")
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)

    print(f" Finished Loading Image Data")
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
        )
    
    return model


def evaluate_models(images, labels, runs=10):
    """
    Train and Evaluate the various models initialized in get_models() function
    Averages the performance of each model over <runs> times to get a more reliable sensing of their performance
    Returns a pandas.DataFrame object containing each model's name as the indexes and their training/test loss and accuracies as columns
    Saves the dataframe to an excel sheet by default
    """
    model_dct = get_models()
    performance_df = pd.DataFrame(columns = ["Training Loss", "Test Loss", "Training Accuracy", "Test Accuracy"],
                                  index=[i for i in model_dct])
    
    performance_df.to_excel("./model_performances.xlsx", sheet_name="model_performances")
    
    model_count = 0
    model_num = len(model_dct)
    for model_name in model_dct:
        model = model_dct[model_name]
        model_count += 1

        training_loss = 0
        training_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        print(f"Fitting and Evaluating Model {model_name} -> Model {model_count} out of {model_num}")
        for i in range(runs):
            print(f"Model {model_count}: Run {i+1}/{runs}")
        
            x_train, x_test, y_train, y_test = train_test_split(
                np.array(images), np.array(labels), test_size=TEST_SIZE
            )
            # Only log one the first and last runs to reduce clutter
            verbose = 2 if i == runs - 1 else 0
            training_metrics = model.fit(x_train, y_train, epochs=EPOCHS, verbose=verbose) 
            training_loss += training_metrics.history["loss"][-1]
            training_accuracy += training_metrics.history["accuracy"][-1]

            curr_test_loss, curr_test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)
            test_loss += curr_test_loss
            test_accuracy += curr_test_accuracy

        performance_df.loc[model_name, 'Training Loss'] = training_loss / runs
        performance_df.loc[model_name, 'Test Loss'] = test_loss / runs
        performance_df.loc[model_name, 'Training Accuracy'] = training_accuracy / runs
        performance_df.loc[model_name, 'Test Accuracy'] = test_accuracy / runs
        print(f" Finished evaluating Model {model_name}\n")
        if len(sys.argv) == 3:
            filename = sys.argv[2]
            model.save(filename)
            print(f"Model saved to {filename}.")
    performance_df.to_excel("./model_performances.xlsx", sheet_name="model_performances")
    return performance_df


def get_models():
    '''
    Return a dictionary of convolutional neural network models with varying layer types and number of layers
    Creates convolutional neural networks with varying:
        1. Number of convolutional layers
        2. Filter number and size
        3. Pooling size
        4. Hidden Layers
        5. Dropout
    Naming for the neural networks is as follows:
    {number of convolutional layers spelt out}_conv_{number of filters}_{filter size}_filter_{pooling size}_pooling_{number of hidden layers}_hidden_{w/wo}_dropout
    '''
    models = {}

    ### MODELS WITH ONE CONVOLUTIONAL LAYER and 32 3x3 Filters ###
    
    models["one_conv_32_3x3_filter_2x2_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["one_conv_32_3x3_filter_4x4_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["one_conv_32_3x3_filter_2x2_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["one_conv_32_3x3_filter_4x4_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["one_conv_32_3x3_filter_2x2_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_32_3x3_filter_4x4_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_32_3x3_filter_2x2_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_32_3x3_filter_4x4_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_32_3x3_filter_2x2_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_32_3x3_filter_4x4_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    ### MODELS WITH ONE CONVOLUTIONAL LAYER and 64 3x3 Filters ###
    models["one_conv_64_3x3_filter_2x2_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_4x4_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_2x2_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_4x4_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_2x2_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_4x4_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_2x2_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_4x4_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_2x2_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["one_conv_64_3x3_filter_4x4_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    ### MODELS WITH TWO CONVOLUTIONAL LAYERS and 32 3x3 Filters ###
    models["two_conv_32_3x3_filter_2x2_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["two_conv_32_3x3_filter_4x4_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["two_conv_32_3x3_filter_2x2_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["two_conv_32_3x3_filter_4x4_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    models["two_conv_32_3x3_filter_2x2_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_32_3x3_filter_4x4_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_32_3x3_filter_2x2_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_32_3x3_filter_4x4_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_32_3x3_filter_2x2_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_32_3x3_filter_4x4_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])
    
    ### MODELS WITH TWO CONVOLUTIONAL LAYER and 64 3x3 Filters ###
    models["two_conv_64_3x3_filter_2x2_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_4x4_pooling_0_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_2x2_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_4x4_pooling_256_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_2x2_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_4x4_pooling_256_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_2x2_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_4x4_pooling_512_hidden_wo_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_2x2_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    models["two_conv_64_3x3_filter_4x4_pooling_512_hidden_w_dropout"] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE), 
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")])

    for model_name in models:
        models[model_name].compile(optimizer="adam",
                                   loss="categorical_crossentropy",
                                   metrics="accuracy")
        models[model_name].summary()
    return models
 



if __name__ == "__main__":
    main()
