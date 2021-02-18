import os
import pandas as pd


def load_preprocessed_data(data_path, preprocessed):
    train_path = 'train.txt'
    val_path = 'validation.txt'
    test_path = 'test.txt'
    if preprocessed:
        train_path = 'train_preprocessed.txt'
        val_path = 'val_preprocessed.txt'
        test_path = 'test_preprocessed.txt'

    training_data = pd.read_csv(os.path.join(data_path, 'train_preprocessed_no_emojis.txt'),
                                sep = ',', header=None)
    test_data = pd.read_csv(os.path.join(data_path, 'test_preprocessed_no_emojis.txt'),
                            sep = ',', header=None)
    val_data = pd.read_csv(os.path.join(data_path, 'val_preprocessed_no_emojis.txt'),
                            sep = ',', header=None)
    training_data.columns = ['Id', 'Latitude', 'Longitude', 'Tweet']
    val_data.columns = ['Id', 'Latitude', 'Longitude', 'Tweet']
    test_data.columns = ['Id', 'Tweet']

    return training_data.dropna().reset_index(drop=True), val_data.dropna().reset_index(drop=True), test_data.dropna().reset_index(drop=True)


