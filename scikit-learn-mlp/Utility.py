import numpy as np
from numpy import linalg as LA

def vector_magnitude(vector):
    #vector = np.square(vector)
    #s = vector.sum()
    #return np.sqrt(s)
    return LA.norm(vector)

def get_file_directory(file_path):
    l1 = file_path.rindex('/')
    l2 = file_path[:l1].rindex('/')
    return file_path[l2+1: l1]


def calculate_accuracy(predictions, actual):
    correct = 0
    incorrect = 0

    for i in range(0, len(predictions)):
        if (predictions[i] == actual[i]).all():
            correct += 1
        else:
            incorrect += 1

    print "  correct: ", correct, "\nincorrect: ", incorrect , ' accuracy: ', correct/float(correct+incorrect)

def calculate_max_probability(probabilities):
    res = np.zeros(len(probabilities))
    res[probabilities.argmax()] = 1
    return res

def adjust_feature_name(_feature_layer):
    feature_layer = ""
    lst_feature_layer = []

    for i, s in enumerate(_feature_layer):
        feature_layer = feature_layer + s
        lst_feature_layer += ["." + s + ".txt"]

        if i != len(_feature_layer)-1:
            feature_layer+="+"

    return feature_layer, lst_feature_layer
