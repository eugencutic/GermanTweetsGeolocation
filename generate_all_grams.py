import numpy as np


def get_grams_util(alphabet, n, grams):
    if n == 0:
        grams[0] = []
        return []
    if n == 1:
        grams[1] = list(alphabet)
        return list(alphabet)
    
    if len(grams[n - 1]) > 0:
        prev_grams = grams[n - 1]
    else:
        prev_grams = get_grams_util(alphabet, n - 1, grams)

    gram = []
    for letter in alphabet:
        for string in prev_grams:
            gram.append(letter + string)
    
    grams[n] = gram
    return gram


def get_grams(alphabet, n):
    grams = [[] for _ in range(n + 1)]
    get_grams_util(alphabet, n, grams)
    return grams
