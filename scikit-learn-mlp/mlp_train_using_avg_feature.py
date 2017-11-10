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


def get_global_AVG_features_training_matrix(output_features, features_directory, class_index, features_used=['.fc8.txt']):
    lines = []
    ret_x = []
    ret_y = []

    with open(output_features) as f:
        lines = f.readlines()

    old_directory_name = ''
    avg_feature_vector = []
    lst_features_tmp = []
    n = 1.0

    for line in lines:
        line = features_directory+line.rstrip()
        file_dir = utl.get_file_directory(line)
        features_vector = []

        for f in features_used:
            with open(line + f) as fil:
                features_vector += map(lambda x: float(x), fil.readline().split(',')[5:])


        if file_dir != old_directory_name:
            if avg_feature_vector != []:
                ret_x.append(avg_feature_vector/float(n))
                #ret_x.append(lst_features_tmp[int(n/2)])
                ret_y.append(one_hot_encoding_class(line[85:line[85:].index('_') + 85], #85, nada: 73
                                                                    class_index))
            #lst_features_tmp = []
            n = 1.0
            old_directory_name = file_dir
            avg_feature_vector = np.array(features_vector)
        else:
            avg_feature_vector = avg_feature_vector + features_vector
            n += 1

        #lst_features_tmp.append(features_vector)

    ret_x.append(avg_feature_vector/float(n))
    #ret_x.append(lst_features_tmp[int(n / 2)])

    ret_y.append(one_hot_encoding_class(line[85:line[85:].index('_') + 85],  # 85, nada: 73
                                        class_index))

    return np.array(ret_x), np.array(ret_y)

def make_experiments(num_hidden_layers, _feature_layer, solver, mode, force_replace_features, force_replace_model):

    feature_layer, lst_feature_layer = utl.adjust_feature_name(_feature_layer)

    filename = 'data_test_sampled/model_'+str(num_hidden_layers)+'_trainsize_70_trained_on_global_AVG_features_'+feature_layer+'_'+solver+'.sav'
    clf = None
    x_global_train = None
    y_global_train = None
    x_global_test = None
    y_global_test = None

    print 'AVG Global features, experiment settings:', str(num_hidden_layers), 'hidden units, classify on local features with settings:' + feature_layer + '_' + solver
    print 'loading sample input/output...'

    if force_replace_features or not os.path.isfile("data_test_sampled/y_global_AVG_"+feature_layer+".txt"):
        output_features = '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_list_prefix.txt'
        features_directory = '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/'

        class_index_path = '/home/kasparov092/sources/c3d/v1.0/data/ucf101/ucfTrainTestlist/classInd.txt'
        class_index = {}

        with open(class_index_path) as f:
           for line in f:
               ic = line.split()
               class_index[ic[1]] = int(ic[0])

        x, y = get_global_AVG_features_training_matrix(output_features, features_directory, class_index, lst_feature_layer)

        np.savetxt("data_test_sampled/x_global_AVG_"+feature_layer+".txt", x) # the save is trained on sample of testing
        np.savetxt("data_test_sampled/y_global_AVG_"+feature_layer+".txt", y)
        x_global_train, x_global_test, y_global_train, y_global_test = train_test_split(x, y, test_size=0.30, random_state=1)
        np.savetxt("data_test_sampled/x_global_train_AVG_"+feature_layer+".txt", x_global_train) # the save is trained on sample of testing
        np.savetxt("data_test_sampled/y_global_train_AVG_"+feature_layer+".txt", y_global_train)
        np.savetxt("data_test_sampled/x_global_test_AVG_"+feature_layer+".txt", x_global_test) # the save is trained on sample of testing
        np.savetxt("data_test_sampled/y_global_test_AVG_"+feature_layer+".txt", y_global_test)
    else:
        #exit()
        x_global_train = np.loadtxt("data_test_sampled/x_global_train_AVG_"+feature_layer+".txt")
        y_global_train = np.loadtxt("data_test_sampled/y_global_train_AVG_"+feature_layer+".txt")
        x_global_test = np.loadtxt("data_test_sampled/x_global_test_AVG_"+feature_layer+".txt")
        y_global_test = np.loadtxt("data_test_sampled/y_global_test_AVG_"+feature_layer+".txt")

    mean_global = x_global_train.mean(axis=0)
    std_global = x_global_train.std(axis=0)

    x_global_train_mean0_var1_global_mean = (x_global_train - mean_global) / std_global
    x_global_test_mean0_var1_global_mean = (x_global_test - mean_global) / std_global

    if force_replace_model or not os.path.isfile(filename):
        clf = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=num_hidden_layers, random_state=1)
        clf.fit(x_global_train_mean0_var1_global_mean, y_global_train)
        pickle.dump(clf, open(filename, 'wb'))
    else:
        clf = pickle.load(open(filename, 'rb'))

    if mode == MODE_GLOBAL_GLOBAL:

        print "Using a model trained on global features only, testing with global features:"

        testing_predictions = clf.predict(x_global_test_mean0_var1_global_mean)
        np.savetxt('data_test_sampled/testing_predictions_global_AVG_model_global_AVG_test_'+feature_layer+'.txt', testing_predictions)

        utl.calculate_accuracy(testing_predictions, y_global_test)

    #====================================================================================
    elif mode == MODE_LOCAL_GLOBAL:
        filename = 'data_test_sampled/model_'+str(num_hidden_layers)+'_trainsize_70_trained_on_local_features_'+feature_layer+'_'+solver+'.sav'
        clf2 = pickle.load(open(filename, 'rb'))

        x_local_train = np.loadtxt("data_test_sampled/x_local_train_"+feature_layer+".txt") # the save is trained on sample of testing
        x_global_test_mean0_var1_local_mean= (x_global_test-x_local_train.mean(axis=0))/x_local_train.std(axis=0)

        print "Using a model trained on local features only, testing with global features:"
        testing_predictions = clf2.predict(x_global_test_mean0_var1_local_mean)
        np.savetxt('data_test_sampled/testing_predictions_local_AVG_model_global_AVG_test_'+feature_layer+'.txt', testing_predictions)
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

        filename = 'data_test_sampled/model_'+str(num_hidden_layers)+'_trainsize_70_trained_on_local_features_'+feature_layer+'_'+solver+'.sav'
        clf2 = pickle.load(open(filename, 'rb'))

        x_local_test = (x_local_test-x_local_train.mean(axis=0))/x_local_train.std(axis=0)

        print "Using a model trained on local features only, testing with local features:"
        testing_predictions = clf2.predict(x_local_test)
        utl.calculate_accuracy(testing_predictions, y_local_test)

make_experiments(50, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(50, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(50, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(50, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(50, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(50, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(50, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(50, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(50, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(50, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(50, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(50, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(50, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(50, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(50, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(50, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(50, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(50, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(50, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(50, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
###############################################################################

make_experiments(101, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(101, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(101, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(101, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(101, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(101, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(101, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(101, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(101, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(101, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(101, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(101, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(101, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(101, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(101, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(101, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(101, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(101, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(101, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(101, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
###############################################################################

make_experiments(200, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(200, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(200, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(200, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(200, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(200, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(200, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(200, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(200, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(200, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(200, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(200, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(200, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(200, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(200, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(200, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(200, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(200, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(200, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(200, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
###############################################################################

make_experiments(3000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(3000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(3000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(3000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(3000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(3000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
###############################################################################

make_experiments(4096, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(4096, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(4096, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(4096, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
###############################################################################

make_experiments(5000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(5000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(5000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(5000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(5000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(5000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
###############################################################################
exit()

make_experiments(3000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(3000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(3000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(3000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(3000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(3000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(3000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)


make_experiments(4096, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(4096, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(4096, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(4096, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(4096, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(4096, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(4096, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)

make_experiments(5000, ['fc6'], 'adam', MODE_GLOBAL_GLOBAL, False, True)
make_experiments(5000, ['fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_GLOBAL,False, True)

make_experiments(5000, ['fc6'], 'adam', MODE_GLOBAL_LOCAL, False, True)
make_experiments(5000, ['fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'adam', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'lbfgs', MODE_GLOBAL_LOCAL,False, True)

make_experiments(5000, ['fc6'], 'adam', MODE_LOCAL_GLOBAL, False, True)
make_experiments(5000, ['fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'adam', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc6', 'fc8'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)
make_experiments(5000, ['fc8', 'fc6'], 'lbfgs', MODE_LOCAL_GLOBAL,False, True)

