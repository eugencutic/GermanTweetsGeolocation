from load_data import *

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
import nltk

data_path = './data/'
training_data, val_data, test_data = load_preprocessed_data(data_path, preprocessed=True)

training_ids = training_data['Id'].values
training_latitudes = np.array(training_data['Latitude'].values)
training_longitudes = np.array(training_data['Longitude'].values)
training_tweets = training_data['Tweet'].values.astype(str)

val_ids = val_data['Id'].values
val_latitudes = np.array(val_data['Latitude'].values)
val_longitudes = np.array(val_data['Longitude'].values)
val_tweets = val_data['Tweet'].values.astype(str)

test_ids = test_data['Id'].values
test_tweets = test_data['Tweet'].values.astype(str)


def tf_idf():
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectorizer.fit(training_tweets)

    train_sequences = vectorizer.transform(training_tweets)
    val_sequences = vectorizer.transform(val_tweets)
    test_sequences = vectorizer.transform(test_tweets)


def BOW():
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(ngram_range=(2, 2))
    vectorizer.fit(training_tweets)

    train_sequences = vectorizer.transform(training_tweets)
    val_sequences = vectorizer.transform(val_tweets)
    test_sequences = vectorizer.transform(test_tweets)


def tokenize(character_tokens=False):
    # tokenize
    tokenizer = Tokenizer(num_words=None, char_level=character_tokens, oov_token='UNK')
    tokenizer.fit_on_texts(training_tweets)

    # get sequences
    train_sequences = tokenizer.texts_to_sequences(training_tweets)
    val_sequences = tokenizer.texts_to_sequences(val_tweets)
    test_sequences = tokenizer.texts_to_sequences(test_tweets)

    # get max length of sequence
    max1, max2, max3 = max([len(s) for s in train_sequences]), max([len(s) for s in val_sequences]), max([len(s) for s in test_sequences])
    max_len = max(max1, max2, max3)
    print(max_len)

    # pad sequences
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    val_sequences = pad_sequences(val_sequences, maxlen=max_len, padding='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')

    # as np arrays
    train_sequences = np.array(train_sequences)
    val_sequences = np.array(val_sequences)
    test_sequences = np.array(test_sequences)

    return train_sequences, val_sequences, test_sequences, tokenizer, max_len

# tokenize
train_sequences, val_sequences, test_sequences, tokenizer, max_len = tokenize()


# given some models predict latitude/longitude only and others both at once,
# predictions are generated differently
def save_submission(model, name, test_array, is_multioutput=False):
    if is_multioutput:
        predictions = model.predict(test_array)
    else:
        predictions_lat = model[0].predict(test_array)
        predictions_long = model[1].predict(test_array)

    submission_df = pd.DataFrame(columns=["id", "lat", "long"])
    submission_df['id'] = test_ids

    if is_multioutput:
        submission_df['lat'] = predictions[:, 0]
        submission_df['long'] = predictions[:, 1]
    else:
        submission_df['lat'] = predictions_lat
        submission_df['long'] = predictions_lat

    submission_df.to_csv(f"./data/{name}.txt", index=False)


def get_mse(model, val_array, labels, is_multioutput=False):
    from sklearn import metrics
    predictions = model.predict(val_array)
    mse = metrics.mean_squared_error(labels, predictions) 
    return mse


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def multioutput_model():
    # labels for multioutput model
    train_labels = np.zeros((training_latitudes.shape[0], 2))
    train_labels[:, 0] = np.array(training_latitudes)
    train_labels[:, 1] = np.array(training_longitudes)

    #labels for multioutput models
    val_labels = np.zeros((val_latitudes.shape[0], 2))
    val_labels[:, 0] = np.array(val_latitudes)
    val_labels[:, 1] = np.array(val_longitudes)

    # multi output model
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.kernel_ridge import KernelRidge
    
    # define and fit
    multi_output = MultiOutputRegressor(svm.NuSVR(), n_jobs=-1)
    multi_output.fit(train_sequences, train_labels)

    # get error
    from sklearn import metrics
    predictions = multi_output.predict(val_sequences)
    mse_1 = metrics.mean_squared_error(val_labels[:, 0], predictions[:, 0]) 
    mse_2 = metrics.mean_squared_error(val_labels[:, 1], predictions[:, 1])
    print(mse_1)
    print(mse_2)

    return multi_output

def separate_coord_grid_search():
    # gridsearch on 2 single output models
    from sklearn import metrics
    mse = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)

    Cs = [0.1, 0.01, 0.001]
    nus = [0.3, 0.7, 0.9, 1]
    params = {'C':Cs, 'nu':nus}

    from sklearn.model_selection import GridSearchCV
    svr_lat_grid = GridSearchCV(svm.NuSVR(), params, cv=5, scoring=mse, n_jobs=-1, verbose=10)

    svr_lat_grid.fit(train_sequences, training_latitudes)

    print(svr_lat_grid.best_params_)

    svr_long_grid = GridSearchCV(svm.NuSVR(), params, cv=5, scoring=mse, n_jobs=-1, verbose=10)

    svr_long_grid.fit(train_sequences, training_longitudes)

    print(svr_long_grid.best_params_)


def separate_coord_nu_svr(train_sequences_lat=train_sequences,
                            train_sequences_long=train_sequences,
                            val_sequences_lat=val_sequences,
                            val_sequences_long=val_sequences,
                            training_latitudes=training_latitudes,
                            training_longitudes=training_longitudes,
                            val_latitudes=val_latitudes,
                            val_longitudes=val_longitudes,
                            test_sequences=test_sequences,
                            sub_name='separate_coord_nu_svr'):
    # separate svr for each coordinate
    svr_lat = svm.NuSVR(C=0.1, nu=0.3, verbose=10)
    svr_lat.fit(train_sequences_lat, training_latitudes)
    mse_lat = get_mse(svr_lat, val_sequences_lat, val_latitudes, is_multioutput=False)

    svr_long = svm.NuSVR(C=0.001, nu=0.7, verbose=10)
    svr_long.fit(train_sequences_long, training_longitudes)
    mse_long = get_mse(svr_long, val_sequences_long, val_longitudes, is_multioutput=False)

    print(mse_lat)
    print(mse_long)
    print((mse_lat + mse_long)/2)

    return svr_lat, svr_long
    


def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'rb')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        line = line.decode('utf-8')
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding


def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 300))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return np.nan_to_num(weight_matrix)


def bilstm(model_path = './best_lstm_model_2.hdf5'):
    from tensorflow.keras.callbacks import ModelCheckpoint

    train_sequences, val_sequences, test_sequences, tokenizer, input_size = tokenize(character_tokens=False)

    embedding_size = 300
    vocab_size = len(tokenizer.word_index)

    # load embedding from file
    raw_embedding = load_embedding('./data/clean_embedding.txt')
    # get vectors in the right order
    embedding_weigths = get_weight_matrix(raw_embedding, tokenizer.word_index)

    # resume training if the model exists
    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
    else:
        model = tensorflow.keras.Sequential([
            layers.Input(shape=(input_size,), name='input', dtype='int64'),
            layers.Embedding(vocab_size+1, embedding_size, input_length=input_size, weights=[embedding_weigths], trainable=True),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Flatten(),
            layers.Dense(2, activation=tensorflow.keras.activations.relu)
        ])

    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=[tensorflow.keras.losses.MeanSquaredError()])

    # labels for multiouput
    train_labels = np.zeros((training_latitudes.shape[0], 2))
    train_labels[:, 0] = np.array(training_latitudes)
    train_labels[:, 1] = np.array(training_longitudes)

    # save best model
    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1,
                                save_best_only=True, mode='auto', period=1)

    # train and return history
    return model.fit(train_sequences, train_labels, validation_split=0.1, 
                    epochs=50, batch_size=128, shuffle=True, callbacks=[checkpoint])


def char_level_CNN(model_path = './best_cnn_model_2.hdf5'):
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import ModelCheckpoint

    # get character tokens and generate sequences
    train_sequences, val_sequences, test_sequences, tokenizer, input_size = tokenize(character_tokens=True)

    # initialize embeddings as one-hot encoded vectors 
    chars = np.array([tokenizer.word_index[word] for word in tokenizer.word_index])
    embedding_weigths = to_categorical(chars).T

    embedding_size = len(chars)
    vocab_size = len(chars)

    # resume trains
    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
    else:
        model = tensorflow.keras.Sequential([
            layers.Input(shape=(input_size,), name='input', dtype='int64'),
            layers.Embedding(vocab_size+1, embedding_size, input_length=input_size, weights=[embedding_weigths]),
            layers.Conv1D(256, 7, activation=tensorflow.keras.activations.relu),
            layers.MaxPooling1D(3),
            layers.Conv1D(256, 7, activation=tensorflow.keras.activations.relu),
            layers.MaxPooling1D(3),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.Dropout(0.5),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.Dropout(0.5),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.Dropout(0.5),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.Dropout(0.5),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.Dropout(0.5),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.Dropout(0.5),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.Dropout(0.5),
            layers.Conv1D(256, 3, activation=tensorflow.keras.activations.relu),
            layers.MaxPooling1D(3),
            layers.Flatten(),
            layers.Dense(1024, activation=tensorflow.keras.activations.relu),
            layers.Dense(1024, activation=tensorflow.keras.activations.relu),
            layers.Dense(2, activation=tensorflow.keras.activations.relu)
        ])

    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=[tensorflow.keras.losses.MeanSquaredError()])

    # multiouput labels
    train_labels = np.zeros((training_latitudes.shape[0], 2))
    train_labels[:, 0] = np.array(training_latitudes)
    train_labels[:, 1] = np.array(training_longitudes)

    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1,
                                save_best_only=True, mode='auto', period=1)

    return model.fit(train_sequences, train_labels, validation_split=0.1, 
                    epochs=100, batch_size=128, shuffle=True, callbacks=[checkpoint])


# if a neural net is trained, then the model is saved, otherwise it is returned
def train_model(model_name):
    if model_name == 'char_cnn':
        history = char_level_CNN()
        plot_history(history)
    if model_name == 'bilstm':
        history = bilstm()
        plot_history(history)
    if model_name == 'multioutput':
        tf_idf()
        return multioutput_model()
    if model_name == 'separate_single_output':
        return separate_coord_nu_svr()


def test_nn(model_path):
    model = load_model(model_path, compile=False)
    train_sequences, val_sequences, test_sequences, tokenizer, input_size = tokenize(character_tokens=False)
    predictions = model.predict(val_sequences)
    from sklearn import metrics
    mse_1 = metrics.mean_squared_error(val_latitudes, predictions[:, 0]) 
    mse_2 = metrics.mean_squared_error(val_longitudes, predictions[:, 1])
    print(mse_1)
    print(mse_2)
    print((mse_1 + mse_2) / 2)
    save_submission(model, 'char_cnn', test_sequences, True)

from tensorflow.keras.utils import plot_model
model = load_model('./best_cnn_model.hdf5', compile=False)
plot_model(model)
