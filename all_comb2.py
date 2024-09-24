# ml_model.py

import numpy as np
import librosa
import os
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from tensorflow.keras.utils import to_categorical
from keras.layers import add, Conv2D, Input, BatchNormalization, TimeDistributed, Embedding, LSTM, GRU, Dense, MaxPooling1D, Dropout, LeakyReLU, ReLU, Flatten, Bidirectional
from keras.models import Model
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint

def add_noise(data, x):
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise

def shift(data, x):
    return np.roll(data, x)

def stretch(data, rate):
    data = librosa.effects.time_stretch(data, rate)
    return data

def validateModel(X_val):
    Model_Loaded = load_model('best_model_22.h5')
    yhat_probs = Model_Loaded.predict(X_val, verbose=1)
    yhat_probs = yhat_probs.reshape(yhat_probs.shape[0], yhat_probs.shape[2])
    yhat_classes = np.argmax(yhat_probs, axis=1)
    return yhat_classes

def evalModel(y_test, y_pred):
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    MatthewsCorrCoef = matthews_corrcoef(y_test, y_pred)

    matrix = classification_report(y_test, y_pred)
    print(matrix)

    return {
           "Accuracy": accuracy,
           "Precision": precision,
           "Recall": recall,
           "F1 score": f1,
           "Cohens kappa": kappa,
           "Matthews correlation coefficient": MatthewsCorrCoef
    }

def InstantiateModel(in_):
    model_2_1 = GRU(32, return_sequences=True, activation=None, go_backwards=True)(in_)
    model_2 = LeakyReLU()(model_2_1)
    model_2 = GRU(128, return_sequences=True, activation=None, go_backwards=True)(model_2)
    model_2 = LeakyReLU()(model_2)
    
    model_3 = GRU(64, return_sequences=True, activation=None, go_backwards=True)(in_)
    model_3 = LeakyReLU()(model_3)
    model_3 = GRU(128, return_sequences=True, activation=None, go_backwards=True)(model_3)
    model_3 = LeakyReLU()(model_3)
    
    model_add_1 = add([model_3, model_2])
    
    model_5 = GRU(128, return_sequences=True, activation=None, go_backwards=True)(model_add_1)
    model_5 = LeakyReLU()(model_5)
    model_5 = GRU(32, return_sequences=True, activation=None, go_backwards=True)(model_5)
    model_5 = LeakyReLU()(model_5)
    
    model_6 = GRU(64, return_sequences=True, activation=None, go_backwards=True)(model_add_1)
    model_6 = LeakyReLU()(model_6)
    model_6 = GRU(32, return_sequences=True, activation=None, go_backwards=True)(model_6)
    model_6 = LeakyReLU()(model_6)
    
    model_add_2 = add([model_5, model_6, model_2_1])
    
    model_7 = Dense(64, activation=None)(model_add_2)
    model_7 = LeakyReLU()(model_7)
    model_7 = Dropout(0.2)(model_7)
    model_7 = Dense(16, activation=None)(model_7)
    model_7 = LeakyReLU()(model_7)
    
    model_9 = Dense(32, activation=None)(model_add_2)
    model_9 = LeakyReLU()(model_9)
    model_9 = Dropout(0.2)(model_9)
    model_9 = Dense(16, activation=None)(model_9)
    model_9 = LeakyReLU()(model_9)
    
    model_add_3 = add([model_7, model_9])

    model_10 = Dense(16, activation=None)(model_add_3)
    model_10 = LeakyReLU()(model_10)
    model_10 = Dropout(0.5)(model_10)
    model_10 = Dense(6, activation="softmax")(model_10)
    
    return model_10

def InstantiateAttributes(dir_):
    X_=[]
    y_=[]
    COPD=[]
    copd_count=0
    for soundDir in os.listdir(dir_):
        if soundDir[-3:] == 'wav' and soundDir[:3] != '103' and soundDir[:3] != '108' and soundDir[:3] != '115':
            p = list(data[data['patient_id'] == int(soundDir[:3])]['disease'])[0]
            if p == 'COPD':
                if soundDir[:3] in COPD and copd_count < 2:
                    data_x, sampling_rate = librosa.load(dir_ + soundDir, res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T, axis=0) 
                    COPD.append(soundDir[:3])
                    copd_count += 1
                    X_.append(mfccs)
                    y_.append(list(data[data['patient_id'] == int(soundDir[:3])]['disease'])[0])
                if soundDir[:3] not in COPD:
                    data_x, sampling_rate = librosa.load(dir_ + soundDir, res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T, axis=0) 
                    COPD.append(soundDir[:3])
                    copd_count = 0
                    X_.append(mfccs)
                    y_.append(list(data[data['patient_id'] == int(soundDir[:3])]['disease'])[0])
                
            if p != 'COPD':
                data_x, sampling_rate = librosa.load(dir_ + soundDir, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T, axis=0) 
                X_.append(mfccs)
                y_.append(list(data[data['patient_id'] == int(soundDir[:3])]['disease'])[0])
            
                data_noise = add_noise(data_x, 0.005)
                mfccs_noise = np.mean(librosa.feature.mfcc(y=data_noise, sr=sampling_rate, n_mfcc=40).T, axis=0) 
                X_.append(mfccs_noise)
                y_.append(p)

                data_shift = shift(data_x, 1600)
                mfccs_shift = np.mean(librosa.feature.mfcc(y=data_shift, sr=sampling_rate, n_mfcc=40).T, axis=0) 
                X_.append(mfccs_shift)
                y_.append(p)

                data_stretch = stretch(data_x, 1.2)
                mfccs_stretch = np.mean(librosa.feature.mfcc(y=data_stretch, sr=sampling_rate, n_mfcc=40).T, axis=0) 
                X_.append(mfccs_stretch)
                y_.append(p)
                
                data_stretch_2 = stretch(data_x, 0.8)
                mfccs_stretch_2 = np.mean(librosa.feature.mfcc(y=data_stretch_2, sr=sampling_rate, n_mfcc=40).T, axis=0) 
                X_.append(mfccs_stretch_2)
                y_.append(p)

    X_ = np.array(X_)
    y_ = np.array(y_)
    
    return X_, y_

def trainModel(X, y):
    K.clear_session()
    batch_size = X.shape[0]
    time_steps = X.shape[1]
    data_dim = X.shape[2]

    Input_Sample = Input(shape=(time_steps, data_dim))
    Output_ = InstantiateModel(Input_Sample)
    Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)

    Model_Enhancer.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adamax())

    ES = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=200, verbose=1, mode='auto', baseline=None,
                       restore_best_weights=False)
    MC = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='auto', verbose=0, save_best_only=True)

    ModelHistory = Model_Enhancer.fit(X, y, batch_size=batch_size, epochs=num_epochs,
                                      validation_split=0.2,
                                      callbacks=[MC],
                                      verbose=1)

if __name__ == "__main__":
    dir_path = "C:/Users/adabs/OneDrive/Desktop/ICBHI_final_database/ICBHI_final_database/"
    X, y = InstantiateAttributes(dir_path)

    # Define the number of epochs for training
    num_epochs = 100

    # Train the model
    trainModel(X, y)

    # Further validation, evaluation, and predictions can be done here
