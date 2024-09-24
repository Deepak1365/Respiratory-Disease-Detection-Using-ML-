import numpy as np
import librosa
import os
from featureExtraction import InstantiateAttributes
from validate import validateModel
from evaluate import evalModel
from model import InstantiateModel
from train import trainModel
from Augmentation import add_noise, shift, stretch
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import add, Conv2D, Input, BatchNormalization, TimeDistributed, Embedding, LSTM, GRU, Dense, MaxPooling1D, Dropout, LeakyReLU, ReLU, Flatten, concatenate, Bidirectional
from keras.models import Model
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint
dir_=r"C:/Users/adabs/OneDrive/Desktop/ICBHI_final_database/ICBHI_final_database/"

# Function to add noise to data
def add_noise(data, x):
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise

# Function to shift data
def shift(data, x):
    return np.roll(data, x)

# Function to stretch data
def stretch(data, rate):
    data = librosa.effects.time_stretch(data, rate)
    return data

# Function to load and preprocess data
def instantiate_attributes(dir_):
    X_ = []
    y_ = []
    # Your data loading and preprocessing logic here
    return X_, y_

# Function to instantiate model architecture
def instantiate_model(in_):
    # Your model architecture definition here
    return model



# Function to validate the model
def validate_model(X_val):
    # Your model validation logic here
    return yhat_classes

# Function to evaluate the model
def evaluate_model(y_test, y_pred):
    # Your model evaluation logic here
    return evaluation_metrics

# Main function
def main():
    dir_ = r"C:/Users/adabs/OneDrive/Desktop/ICBHI_final_database/ICBHI_final_database/"

    # Load and preprocess data
    X, y = instantiate_attributes(dir_)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = instantiate_model(input_shape)

    # Train model
    trained_model = trainModel(X_train, y_train)

    # Validate model
    yhat_classes = validate_model(X_test)

    # Evaluate model
    evaluation_metrics = evaluate_model(y_test, yhat_classes)

    print("Evaluation Metrics:", evaluation_metrics)

if __name__ == "__main__":
    main()





