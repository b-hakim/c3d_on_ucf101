import numpy as np
import pickle
from sklearn import preprocessing


def one_hot_encoding_class(i):
    out = np.zeros(101)
    out[i-1] = 1
    return out


def avg_class(class_id, samples, samples_output):
    class_hot_encoded = np.zeros(101)
    class_hot_encoded[class_id-1] = 1
    indices = [i for i, x in enumerate(samples_output) if (x == class_hot_encoded).all()]
    return np.average([samples[indices]], axis=1)


def get_new_test_cases(samples, samples_output):
    testing_samples = []
    testing_samples_output = []
    samples = np.array(map(lambda x: preprocessing.scale(x), samples))

    for i in range(1, 102):
        testing_samples.append(avg_class(i, samples, samples_output)[0])
        testing_samples_output.append(one_hot_encoding_class(i))

    print 'loading model'
    filename = 'model_3000.sav'
    clf = pickle.load(open(filename, 'rb'))

    print 'predicting the average of 101 avg classes'
    testing_samples[0] += [1]
    testing_predictions = clf.predict(testing_samples)

    correct = 0
    incorrect = 0

    for i in range(0, len(testing_predictions)):
        if (testing_predictions[i] == testing_samples_output[i]).all():
            correct += 1
        else:
            incorrect += 1

    print "  correct: ", correct, "\nincorrect: ", incorrect

x = np.loadtxt("x.txt")
y = np.loadtxt("y.txt")

get_new_test_cases(x, y)