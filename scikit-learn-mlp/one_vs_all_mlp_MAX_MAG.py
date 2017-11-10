from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
from sklearn import preprocessing
import Utility as utl
def transform_output_to_1_0(class_index, output_hot_coded):
    output = []

    for o in output_hot_coded:
        ind = o.tolist().index(1)

        if class_index == ind:
            output.append(1)
        else:
            output.append(0)

    return output

#lst_probabilities = np.loadtxt('data_test_sampled/101_classifiers_probabilities_global_MAX_MAG.txt')
#y_global_test = np.loadtxt("data_test_sampled/y_global_test_MAX_MAG.txt")
#y_local_test = np.loadtxt("data_test_sampled/y_local_test.txt")
#lst_predictions = []

#for prob in lst_probabilities:
#    lst_predictions.append(utl.calculate_max_probability(prob))

#utl.calculate_accuracy(lst_predictions, y_local_test)
#exit()

#======================================================================================
print 'Using a model trained on local features only, testing with global features:'
x_global_train = np.loadtxt("data_test_sampled/x_global_train_MAX_MAG.txt")
y_global_train = np.loadtxt("data_test_sampled/y_global_train_MAX_MAG.txt")
x_global_test = np.loadtxt("data_test_sampled/x_global_test_MAX_MAG.txt")
y_global_test = np.loadtxt("data_test_sampled/y_global_test_MAX_MAG.txt")

x_local_train = np.loadtxt("data_test_sampled/x_local_train.txt") # the save is trained on sample of testing
y_local_train = np.loadtxt("data_test_sampled/y_local_train.txt")
x_local_test = np.loadtxt("data_test_sampled/x_local_test.txt") # the save is trained on sample of testing
y_local_test = np.loadtxt("data_test_sampled/y_local_test.txt")

mean = x_local_train.mean(axis=0)
std = x_local_train.std(axis=0)
# #mean = x_global_train.mean(axis=0)
#std = x_global_train.std(axis=0)

x_global_train = (x_global_train-mean)/std
x_global_test = (x_global_test-mean)/std
x_local_train = (x_local_train-mean)/std
x_local_test = (x_local_test-mean)/std

lst_predictions = []
lst_probabilities = []

for class_index in range(0, 101):
    print "fitting/predicting with classifier: ", str(class_index)
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=3000, random_state=1)

    clf.fit(x_local_train, transform_output_to_1_0(class_index, y_local_train))

    #filename = 'data_test_sampled/101_classifiers/model_'+str(class_index)+'.sav'
    #pickle.dump(clf, open(filename, 'wb'))

    #lst_predictions += [clf.predict(x_global_test)]
    lst_probabilities += [clf.predict_proba(x_global_test)[:, 1]]


#lst_predictions = np.transpose(lst_predictions)
lst_probabilities = np.transpose(lst_probabilities)
#print lst_predictions
print lst_probabilities[0]

#np.savetxt('data_test_sampled/101_classifiers_predictions_global_MAX_MAG.txt', lst_predictions)
np.savetxt('data_test_sampled/101_classifiers_probabilities_trained_local_tested_global_MAX_MAG.txt', lst_probabilities)

lst_predictions = []

for prob in lst_probabilities:
    lst_predictions.append(utl.calculate_max_probability(prob))

utl.calculate_accuracy(lst_predictions, y_global_test)

#======================================================================================
print 'Using a model trained on global features only, testing with local features:'
x_global_train = np.loadtxt("data_test_sampled/x_global_train_MAX_MAG.txt")
y_global_train = np.loadtxt("data_test_sampled/y_global_train_MAX_MAG.txt")
x_global_test = np.loadtxt("data_test_sampled/x_global_test_MAX_MAG.txt")
y_global_test = np.loadtxt("data_test_sampled/y_global_test_MAX_MAG.txt")

x_local_train = np.loadtxt("data_test_sampled/x_local_train.txt") # the save is trained on sample of testing
y_local_train = np.loadtxt("data_test_sampled/y_local_train.txt")
x_local_test = np.loadtxt("data_test_sampled/x_local_test.txt") # the save is trained on sample of testing
y_local_test = np.loadtxt("data_test_sampled/y_local_test.txt")

mean = x_global_train.mean(axis=0)
std = x_global_train.std(axis=0)

x_global_train = (x_global_train-mean)/std
x_global_test = (x_global_test-mean)/std
x_local_train = (x_local_train-mean)/std
x_local_test = (x_local_test-mean)/std

lst_predictions = []
lst_probabilities = []

for class_index in range(0, 101):
    print "fitting/predicting with classifier: ", str(class_index)
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=3000, random_state=1)

    clf.fit(x_global_train, transform_output_to_1_0(class_index, y_global_train))

    #filename = 'data_test_sampled/101_classifiers/model_'+str(class_index)+'.sav'
    #pickle.dump(clf, open(filename, 'wb'))

    #lst_predictions += [clf.predict(x_global_test)]
    lst_probabilities += [clf.predict_proba(x_local_test)[:, 1]]


#lst_predictions = np.transpose(lst_predictions)
lst_probabilities = np.transpose(lst_probabilities)
#print lst_predictions
print lst_probabilities[0]

#np.savetxt('data_test_sampled/101_classifiers_predictions_global_MAX_MAG.txt', lst_predictions)
np.savetxt('data_test_sampled/101_classifiers_probabilities_trained_global_tested_local.txt', lst_probabilities)

lst_predictions = []

for prob in lst_probabilities:
    lst_predictions.append(utl.calculate_max_probability(prob))

utl.calculate_accuracy(lst_predictions, y_local_test)

#======================================================================================
print 'Using a model trained on global features only, testing with global features:'
x_global_train = np.loadtxt("data_test_sampled/x_global_train_MAX_MAG.txt")
y_global_train = np.loadtxt("data_test_sampled/y_global_train_MAX_MAG.txt")
x_global_test = np.loadtxt("data_test_sampled/x_global_test_MAX_MAG.txt")
y_global_test = np.loadtxt("data_test_sampled/y_global_test_MAX_MAG.txt")

x_local_train = np.loadtxt("data_test_sampled/x_local_train.txt") # the save is trained on sample of testing
y_local_train = np.loadtxt("data_test_sampled/y_local_train.txt")
x_local_test = np.loadtxt("data_test_sampled/x_local_test.txt") # the save is trained on sample of testing
y_local_test = np.loadtxt("data_test_sampled/y_local_test.txt")

mean = x_global_train.mean(axis=0)
std = x_global_train.std(axis=0)

x_global_train = (x_global_train-mean)/std
x_global_test = (x_global_test-mean)/std
x_local_train = (x_local_train-mean)/std
x_local_test = (x_local_test-mean)/std

lst_predictions = []
lst_probabilities = []

for class_index in range(0, 101):
    print "fitting/predicting with classifier: ", str(class_index)
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=3000, random_state=1)

    clf.fit(x_global_train, transform_output_to_1_0(class_index, y_global_train))

    #filename = 'data_test_sampled/101_classifiers/model_'+str(class_index)+'.sav'
    #pickle.dump(clf, open(filename, 'wb'))

    #lst_predictions += [clf.predict(x_global_test)]
    lst_probabilities += [clf.predict_proba(x_global_test)[:, 1]]


#lst_predictions = np.transpose(lst_predictions)
lst_probabilities = np.transpose(lst_probabilities)
#print lst_predictions
print lst_probabilities[0]

#np.savetxt('data_test_sampled/101_classifiers_predictions_global_MAX_MAG.txt', lst_predictions)
np.savetxt('data_test_sampled/101_classifiers_probabilities_trained_global_tested_global.txt', lst_probabilities)

lst_predictions = []

for prob in lst_probabilities:
    lst_predictions.append(utl.calculate_max_probability(prob))

utl.calculate_accuracy(lst_predictions, y_global_test)

#======================================================================================
print 'Using a model trained on local features only, testing with local features:'

x_local_train = np.loadtxt("data_test_sampled/x_local_train.txt") # the save is trained on sample of testing
y_local_train = np.loadtxt("data_test_sampled/y_local_train.txt")
x_local_test = np.loadtxt("data_test_sampled/x_local_test.txt") # the save is trained on sample of testing
y_local_test = np.loadtxt("data_test_sampled/y_local_test.txt")

mean = x_local_train.mean(axis=0)
std = x_local_train.std(axis=0)

x_local_train = (x_local_train-mean)/std
x_local_test = (x_local_test-mean)/std

lst_predictions = []
lst_probabilities = []

for class_index in range(0, 101):
    print "fitting/predicting with classifier: ", str(class_index)
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=3000, random_state=1)

    clf.fit(x_local_train, transform_output_to_1_0(class_index, y_local_train))

    #filename = 'data_test_sampled/101_classifiers/model_'+str(class_index)+'.sav'
    #pickle.dump(clf, open(filename, 'wb'))

    #lst_predictions += [clf.predict(x_global_test)]
    lst_probabilities += [clf.predict_proba(x_local_test)[:, 1]]


#lst_predictions = np.transpose(lst_predictions)
lst_probabilities = np.transpose(lst_probabilities)
#print lst_predictions
print lst_probabilities[0]

#np.savetxt('data_test_sampled/101_classifiers_predictions_global_MAX_MAG.txt', lst_predictions)
np.savetxt('data_test_sampled/101_classifiers_probabilities_trained_global_tested_global.txt', lst_probabilities)

lst_predictions = []

for prob in lst_probabilities:
    lst_predictions.append(utl.calculate_max_probability(prob))

utl.calculate_accuracy(lst_predictions, y_local_test)
