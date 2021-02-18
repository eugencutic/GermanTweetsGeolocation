### The following code is an attempt to compute string kernels
### However, there are no result in this sense, given the 
### large amount of computing power necessary for generating the kernel matrix
### The computed kernel was k 0/1. 
### (as explained in R.Ionescu & M. Gaman - Combining Deep Learning and String Kernels for the Localization of Swiss German Tweets)
### Given 2 words - get the alphabet defined by their unique letters and generate all words up to length n over that alphabet
### For each generated subword (n-gram) add 1 to the similarity of the 2 given words if both contain the subword

import os
import pandas as pd

from generate_all_grams import *

load_data = False

def load_preprocessed_data(data_path):
  training_data = pd.read_csv(os.path.join(data_path, 'train_preprocessed.txt'),
                            sep = ',', header=None)
  test_data = pd.read_csv(os.path.join(data_path, 'test_preprocessed.txt'),
                              sep = ',', header=None)
  val_data = pd.read_csv(os.path.join(data_path, 'val_preprocessed.txt'),
                              sep = ',', header=None)
  training_data.columns = ['Id', 'Latitude', 'Longitude', 'Tweet']
  val_data.columns = ['Id', 'Latitude', 'Longitude', 'Tweet']
  test_data.columns = ['Id', 'Tweet']

  return training_data.dropna().reset_index(drop=True), val_data.dropna().reset_index(drop=True), test_data.dropna().reset_index(drop=True)

data_path = './data/'

if load_data:
    training_data, val_data, test_data = load_preprocessed_data(data_path)

    training_ids = training_data['Id'].values
    training_latitudes = training_data['Latitude'].values
    training_longitudes = training_data['Longitude'].values
    training_tweets = training_data['Tweet'].values

    val_ids = val_data['Id'].values
    val_latitudes = val_data['Latitude'].values
    val_longitudes = val_data['Longitude'].values
    val_tweets = val_data['Tweet'].values

    test_ids = test_data['Id'].values
    test_tweets = test_data['Tweet'].values

    words = {'OOV'}

    for i, tweet in enumerate(training_tweets):
        print(i)
        words = words | set(tweet.split())

    print(len(words))

    headers = ['words']
    for word in words:
        headers.append(word)
    headers = ','.join(headers)

    with open(os.path.join(data_path, 'kernel.txt'), mode='a', encoding='utf-8') as f:
        f.write(headers)
else:
    start = 0
    with open(os.path.join(data_path, 'kernel.txt'), mode='r', encoding='utf-8') as f:
        words = f.readline().split(',')[1:]
        for _ in f.readlines():
            start += 1

    progress_line = 'Computing kernel line for word '
    progress_word = 'Computing similarity to word '

    n = len(words)

    with open(os.path.join(data_path, 'kernel.txt'), mode='a', encoding='utf-8') as f:
        for i in range(start, n):
            kernel = np.zeros(n)
            for j in range(n):
                os.system('cls')
                print(progress_line + str(i) + ' out of ' + str(n))
                print(progress_word + str(j) + ' out of ' + str(n))
                if words[i] == 'OOV' or words[j] == 'OOV':
                    continue
                else:
                    alphabet = list(set(words[i]) | set(words[j]))
                    all_grams = get_grams(alphabet, 5)
                    for n_gram in all_grams:
                        for gram in n_gram:
                            if gram in words[i] and gram in words[j]:
                                kernel[i] += 1
            line = words[i] + ',' + ','.join([str(j) for k in kernel])
            f.write(line)

