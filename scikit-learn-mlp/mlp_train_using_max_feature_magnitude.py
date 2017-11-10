from sklearn.neural_network import MLPClassifier
import Utility as utl
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

MODE_GLOBAL_GLOBAL = 0
MODE_GLOBAL_LOCAL = 1
MODE_LOCAL_GLOBAL = 2
MODE_LOCAL_LOCAL = 3

def one_hot_encoding_class(class_name, class_index):
    i = class_index[class_name]
    out = np.zeros(101)
    out[i-1] = 1
    return out


def get_global_MAX_MAG_features_training_matrix(output_features, features_directory,
                                                class_index, features_used=['.fc6.txt'],
                                                min_mag=True,max_mag=True, median_mag=True):
    lines = []
    ret_x = []
    ret_y = []

    with open(output_features) as f:
        lines = f.readlines()

    old_directory_name = ''
    max_feature_vector = []
    # median_mag_vector = []
    min_feature_vector = []
    max_magnitude = -1
    min_magnitude = 100000000000
    S = []
    # fastPointer = 1
    # slowPointer = 0

    for line in lines:
        line = features_directory+line.rstrip()
        file_dir = utl.get_file_directory(line)
        features_vector = []

        for f in features_used:
            with open(line + f) as fil:
                features_vector += map(lambda x: float(x), fil.readline().split(',')[5:])

        if file_dir != old_directory_name:
            if min_mag:
                S += min_feature_vector

            if median_mag and min_feature_vector != [] and max_feature_vector != []:
                S += ((np.array(min_feature_vector)+np.array(max_feature_vector))/2).tolist()

            if max_mag:
                S += max_feature_vector

            if S != []:
                ret_x.append(S)
                ret_y.append(one_hot_encoding_class(line[85:line[85:].index('_') + 85], #85, nada: 73
                                                    class_index))
                S = []

            max_magnitude = utl.vector_magnitude(features_vector)
            min_magnitude = max_magnitude

            max_feature_vector = features_vector
            min_feature_vector = features_vector
            old_directory_name = file_dir
            fastPointer = 1
            slowPointer = 0
            #median_mag_vector = features_vector
        else:
            mag = utl.vector_magnitude(features_vector)

            if mag > max_magnitude:
                max_feature_vector = features_vector
                max_magnitude = mag

            if mag < min_magnitude:
                min_feature_vector = features_vector
                min_magnitude = mag

            # fastPointer += 1
            #
            # if fastPointer % 2 == 0:
            #     slowPointer += 1
            #     median_mag_vector = features_vector

    if min_mag:
        S += min_feature_vector

    if median_mag:
        S += ((np.array(min_feature_vector) + np.array(max_feature_vector)) / 2).tolist()

    if max_mag:
        S += max_feature_vector

    ret_x.append(S)
    ret_y.append(one_hot_encoding_class(line[85:line[85:].index('_') + 85],
                                        class_index))
    return np.array(ret_x), np.array(ret_y)


def make_experiment(num_hidden_layers, _feature_layer, solver, mode, force_replace, min_mag, max_mag):
    feature_layer, lst_feature_layer = utl.adjust_feature_name(_feature_layer)
    if min_mag and max_mag:
        print ("=================Min+Max_Mag===============")
    elif min_mag:
        print("===============MinMag===================")
    elif max_mag:
        print("===============MaxMag=================")

    print str(num_hidden_layers), 'hidden units, settings:' + feature_layer + '_' + solver

    filename = 'data_test_sampled/model_' + str(num_hidden_layers) + '_trainsize_70_trained_on_global_MAX_MAG_features_' + feature_layer + '_' + solver + '.sav'

    clf = None
    x_global_train = None
    y_global_train = None
    x_global_test = None
    y_global_test = None

    if force_replace or not os.path.isfile("data_test_sampled/x_global_train_MAX_MAG_"+feature_layer+".txt"):
        output_features = '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_list_prefix.txt'
        features_directory = '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/'

        class_index_path = '/home/kasparov092/sources/c3d/v1.0/data/ucf101/ucfTrainTestlist/classInd.txt'
        class_index = {}


        print 'loading class indices...'

        with open(class_index_path) as f:
           for line in f:
               ic = line.split()
               class_index[ic[1]] = int(ic[0])

        print 'loading sample input/output...'
        #
        x, y = get_global_MAX_MAG_features_training_matrix(output_features, features_directory, class_index, lst_feature_layer, min_mag,max_mag, False)

        x_global_train, x_global_test, y_global_train, y_global_test = train_test_split(x, y, test_size=0.30, random_state=42)
        np.savetxt("data_test_sampled/x_global_train_MAX_MAG_"+feature_layer+".txt", x_global_train) # the save is trained on sample of testing
        np.savetxt("data_test_sampled/y_global_train_MAX_MAG_"+feature_layer+".txt", y_global_train)
        np.savetxt("data_test_sampled/x_global_test_MAX_MAG_"+feature_layer+".txt", x_global_test) # the save is trained on sample of testing
        np.savetxt("data_test_sampled/y_global_test_MAX_MAG_"+feature_layer+".txt", y_global_test)
        #exit()
    else:
        x_global_train = np.loadtxt("data_test_sampled/x_global_train_MAX_MAG_"+feature_layer+".txt")
        y_global_train = np.loadtxt("data_test_sampled/y_global_train_MAX_MAG_"+feature_layer+".txt")
        x_global_test = np.loadtxt("data_test_sampled/x_global_test_MAX_MAG_"+feature_layer+".txt")
        y_global_test = np.loadtxt("data_test_sampled/y_global_test_MAX_MAG_"+feature_layer+".txt")

    mean_global = x_global_train.mean(axis=0)
    std_global = x_global_train.std(axis=0)
    #std_global[[i for i, x in enumerate(std_global) if x == 0]] = 1

    x_global_train_mean0_var1_global_mean = (x_global_train-mean_global)/std_global
    x_global_test_mean0_var1_global_mean = (x_global_test-mean_global)/std_global

    #x_global_train = preprocessing.scale(x_global_train)
    #x_global_test = preprocessing.scale(x_global_test)
    clf = None

    if force_replace or not os.path.isfile(filename):
        clf = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=num_hidden_layers, random_state=1)
        clf.fit(x_global_train_mean0_var1_global_mean, y_global_train)
        pickle.dump(clf, open(filename, 'wb'))
    else:
        clf = pickle.load(open(filename, 'rb'))

    if mode == MODE_GLOBAL_GLOBAL:
        print "Using a model trained on global features only, testing with global features:"
        testing_predictions = clf.predict(x_global_test_mean0_var1_global_mean)
        #np.savetxt('data_test_sampled/testing_predictions_global_model_global_test.txt', testing_predictions)

        utl.calculate_accuracy(testing_predictions, y_global_test)

    #====================================================================================
    elif mode == MODE_LOCAL_GLOBAL:
        filename2 = 'data_test_sampled/model_'+str(num_hidden_layers)+'_trainsize_70_trained_on_local_features_'+feature_layer+'_'+solver+'.sav'
        clf2 = pickle.load(open(filename2, 'rb'))

        x_local_train = np.loadtxt("data_test_sampled/x_local_train_"+feature_layer+".txt") # the save is trained on sample of testing
        x_global_test_mean0_var1_local_mean = (x_global_test-x_local_train.mean(axis=0))/x_local_train.std(axis=0)

        print "Using a model trained on local features only, testing with global features:"
        testing_predictions = clf2.predict(x_global_test_mean0_var1_local_mean)
        utl.calculate_accuracy(testing_predictions, y_global_test)
    #====================================================================================
    elif mode == MODE_GLOBAL_LOCAL:
        x_local_test = np.loadtxt("data_test_sampled/x_local_test_"+feature_layer+".txt") # the save is trained on sample of testing
        y_local_test = np.loadtxt("data_test_sampled/y_local_test_"+feature_layer+".txt")
        x_local_test = (x_local_test - mean_global)/std_global

        print "Using a model trained on global features only, testing with local features:"
        testing_predictions = clf.predict(x_local_test)
        utl.calculate_accuracy(testing_predictions, y_local_test)
    #====================================================================================
    else:
        x_local_test = np.loadtxt("data_test_sampled/x_local_test_"+feature_layer+".txt") # the save is trained on sample of testing
        y_local_test = np.loadtxt("data_test_sampled/y_local_test_"+feature_layer+".txt")
        x_local_train = np.loadtxt("data_test_sampled/x_local_train_"+feature_layer+".txt") # the save is trained on sample of testing
        x_local_test = (x_local_test-x_local_train.mean(axis=0))/x_local_train.std(axis=0)
        filename = 'data_test_sampled/model_'+str(num_hidden_layers)+'_trainsize_70_trained_on_local_features_'+feature_layer+'_'+solver+'.sav'
        clf2 = pickle.load(open(filename, 'rb'))
        print "Using a model trained on local features only, testing with local features:"
        testing_predictions = clf2.predict(x_local_test)
        utl.calculate_accuracy(testing_predictions, y_local_test)

make_experiment(50, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False)

make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True)
exit()
make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, True)

make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL, True, True, False)
make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL, True, False, True)
#make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL, True, True, True)

make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL, True, True, False)
make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL, True, False, True)
make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL, True, True, True)

################################################################True, #################

make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True,  True, False)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True,  False, True)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True,  True, True)

make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True,  True, False)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True,  False, True)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True,  True, True)

make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True,  True, False)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True,  False, True)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True,  True, True)

################################################################True, #################

make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True,  True, False)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True,  False, True)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True,  True, True)

make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True,  True, False)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True,  False, True)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True,  True, True)

make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True,  True, False)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True,  False, True)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True,  True, True)

################################################################True, #################

make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, False, True)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, True)

make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True , True, False)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True,False, True)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True , True, True)

make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True , True, False)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True , False, True)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True , True, True)

################################################################True, #################

make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, False, True)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, True)

make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, False, True)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, True)

make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, False, True)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, True)

################################################################True, #################

make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, False, True)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, True)

make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, False, True)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, True)

make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, False, True)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, True)

########################################################################################
exit()
print ("==========================50-Global-Global======================================")
print ('MinMag')
make_experiment(50, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, True, False, False)
# make_experiment(50, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(50, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(50, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# print ('MaxMag')
make_experiment(50, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, False, False, True)
# make_experiment(50, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(50, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(50, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# print ('MinMag+AVG+MaxMag')
make_experiment(50, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, True, True, True)
exit()# make_experiment(50, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(50, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(50, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# print ("=========================50-Global-Local===========================")
#
# print ('MinMag')
# make_experiment(50, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False, False)
# make_experiment(50, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
# make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
# make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
# make_experiment(50, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
# make_experiment(50, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
# make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
# make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
# print ('MaxMag')
# make_experiment(50, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, False, False, True)
# make_experiment(50, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
# make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
# make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
# make_experiment(50, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
# make_experiment(50, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
# make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
# make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
# print ('MinMag+AVG+MaxMag')
# make_experiment(50, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True, True, True)
# make_experiment(50, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
# make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
# make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
# make_experiment(50, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
# make_experiment(50, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
# make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
# make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)

print ("=========================50-Local-Global===========================")

print ('MinMag')
make_experiment(50, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, True, True, False, False)
make_experiment(50, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(50, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(50, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
print ('MaxMag')
make_experiment(50, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, False, False, True)
make_experiment(50, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(50, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(50, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
print ('MinMag+AVG+MaxMag')
make_experiment(50, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True, True, True)
make_experiment(50, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(50, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(50, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(50, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(50, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(50, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(50, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)

# print ("==========================101-Global-Global======================================")
# print ('MinMag')
# make_experiment(101, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, True, False, False)
# make_experiment(101, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(101, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(101, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# print ('MaxMag')
# make_experiment(101, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, False, False, True)
# make_experiment(101, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(101, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(101, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# print ('MinMag+MaxMag')
# make_experiment(101, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True, True, True)
# make_experiment(101, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(101, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(101, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)

print ("==========================101-Global-Local======================================")
print ('MinMag')
make_experiment(101, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False, False)
make_experiment(101, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(101, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(101, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
print ('MaxMag')
make_experiment(101, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, False, False, True)
make_experiment(101, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(101, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(101, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(101, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True, True, True)
make_experiment(101, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(101, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(101, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
print ("==========================101-Local-Global======================================")
print ('MinMag')
make_experiment(101, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, True, True, False, False)
make_experiment(101, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(101, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(101, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
print ('MaxMag')
make_experiment(101, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, False, False, True)
make_experiment(101, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(101, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(101, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(101, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True, True, True)
make_experiment(101, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(101, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(101, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(101, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(101, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(101, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(101, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
#
# print ("==========================200======================================")
# print ('MinMag')
# make_experiment(200, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, True, False, False)
# make_experiment(200, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(200, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(200, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# print ('MaxMag')
# make_experiment(200, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, False, False, True)
# make_experiment(200, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(200, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(200, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# print ('MinMag+MaxMag')
# make_experiment(200, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True, True, True)
# make_experiment(200, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(200, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(200, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)


print ("==========================200-Global-Local======================================")
print ('MinMag')
make_experiment(200, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False, False)
make_experiment(200, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(200, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(200, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
print ('MaxMag')
make_experiment(200, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, False, False, True)
make_experiment(200, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(200, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(200, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(200, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True, True, True)
make_experiment(200, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(200, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(200, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)


print ("==========================200-Local-Global======================================")
print ('MinMag')
make_experiment(200, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, True, True, False, False)
make_experiment(200, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(200, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(200, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
print ('MaxMag')
make_experiment(200, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, False, False, True)
make_experiment(200, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(200, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(200, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(200, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True, True, True)
make_experiment(200, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(200, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(200, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(200, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(200, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(200, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(200, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)

# print ("==========================3000======================================")
# print ('MinMag')
# make_experiment(3000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, True, False, False)
# make_experiment(3000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# print ('MaxMag')
# make_experiment(3000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, False, False, True)
# make_experiment(3000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, False, False, True)
# make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, False, False, True)
# make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,True, False, False, True)
# make_experiment(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, False, False, True)
# make_experiment(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, False, False, True)
# make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, False, True, True)
# make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, False, True, True)
# print ('MinMag+MaxMag')
# make_experiment(3000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True, True, True)
# make_experiment(3000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)

print ("==========================3000-Global-Local======================================")
print ('MinMag')
make_experiment(3000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False, False)
make_experiment(3000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
print ('MaxMag')
make_experiment(3000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, False, False, True)
make_experiment(3000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,True, False, False, True)
make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,True, False, False, True)
make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,True, False, False, True)
make_experiment(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, False, False, True)
make_experiment(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, False, False, True)
make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, False, True, True)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, False, True, True)
print ('MinMag+MaxMag')
make_experiment(3000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True, True, True)
make_experiment(3000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)

print ("==========================3000-Local-Global======================================")
print ('MinMag')
make_experiment(3000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, True, True, False, False)
make_experiment(3000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(3000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(3000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
print ('MaxMag')
make_experiment(3000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, True, False, False, True)
make_experiment(3000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,True, False, False, True)
make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,True, False, False, True)
make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,True, False, False, True)
make_experiment(3000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, False, False, True)
make_experiment(3000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, False, False, True)
make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, False, True, True)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, False, True, True)
print ('MinMag+MaxMag')
make_experiment(3000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True, True, True)
make_experiment(3000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(3000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(3000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(3000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(3000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(3000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
#
# print ("==========================4096======================================")
# print ('MinMag')
# make_experiment(4096, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, True, True, False, False)
# make_experiment(4096, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# print ('MaxMag')
# make_experiment(4096, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, False, False, True)
# make_experiment(4096, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# print ('MinMag+MaxMag')
# make_experiment(4096, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True, True, True)
# make_experiment(4096, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)

print ("==========================4096-Global-Local======================================")
print ('MinMag')
make_experiment(4096, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False, False)
make_experiment(4096, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
print ('MaxMag')
make_experiment(4096, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, False, False, True)
make_experiment(4096, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(4096, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True, True, True)
make_experiment(4096, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)

print ("==========================4096-Local-Global======================================")
print ('MinMag')
make_experiment(4096, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, True, True, False, False)
make_experiment(4096, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(4096, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(4096, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
print ('MaxMag')
make_experiment(4096, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, False, False, True)
make_experiment(4096, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(4096, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(4096, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(4096, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True, True, True)
make_experiment(4096, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(4096, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(4096, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(4096, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(4096, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(4096, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)


# print ("==========================5000======================================")
# print ('MinMag')
# make_experiment(5000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False, False)
# make_experiment(5000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,True, True, False, False)
# print ('MaxMag')
# make_experiment(5000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, False, False, True)
# make_experiment(5000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, False, True)
# make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, False, True, True)
# print ('MinMag+MaxMag')
# make_experiment(5000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True, True, True)
# make_experiment(5000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)
# make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True, True, True)

print ("==========================5000-Global-Local======================================")
print ('MinMag')
make_experiment(5000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, True, True, False, False)
make_experiment(5000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,True, True, False, False)
print ('MaxMag')
make_experiment(5000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, False, False, True)
make_experiment(5000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, False, True)
make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(5000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True, True, True)
make_experiment(5000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True, True, True)

print ("==========================5000-Local-Global======================================")
print ('MinMag')
make_experiment(5000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, True, True, False, False)
make_experiment(5000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(5000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(5000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,True, True, False, False)
print ('MaxMag')
make_experiment(5000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, False, False, True)
make_experiment(5000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(5000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(5000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, False, True)
make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, False, True, True)
print ('MinMag+MaxMag')
make_experiment(5000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True, True, True)
make_experiment(5000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(5000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(5000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(5000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(5000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(5000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)
make_experiment(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True, True, True)


