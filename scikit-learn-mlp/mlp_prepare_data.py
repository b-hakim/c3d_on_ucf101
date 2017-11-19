from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import Utility as utl
import os


def one_hot_encoding_class(class_name, class_index):
    i = class_index[class_name]
    out = np.zeros(101)
    out[i-1] = 1
    return out

def get_training_matrix(output_features, features_directory, class_index, features_used=['.fc8.txt']):
    lines = []
    with open(output_features) as f:
        lines = f.readlines()
    passes = 0
    features_file_pathes_used = {}

    for line in lines:
        full_line = features_directory+line.rstrip()
        features_file_pathes_used[full_line] = []

        for f in features_used:
            features_file_pathes_used[full_line] += [f]

    ret_x = []
    ret_y = []

    for feature_set_key in features_file_pathes_used.keys():
        sub_features = []

        for sub_feature in features_file_pathes_used[feature_set_key]:
            with open(feature_set_key + sub_feature) as fil:
                sub_features += map(lambda x: float(x), fil.readline().split(',')[5:])

        if sub_features == []:
            passes += 1
            continue

        ret_x.append(sub_features)
        ret_y.append(one_hot_encoding_class(feature_set_key[85:feature_set_key[85:].index('_')+85],
                                            class_index))
    print 'passs:', passes
    return np.array(ret_x), np.array(ret_y)

def make_experiments(num_hidden_layers, _feature_layer, solver, force_replace, server_path=False):
    feature_layer, lst_feature_layer = utl.adjust_feature_name(_feature_layer)

    print 'experiment settings:', str(num_hidden_layers), 'hidden units, classify on local features with settings:'+feature_layer+'_'+solver
    print 'loading sample input/output...'

    filename = 'data_test_sampled/model_'+str(num_hidden_layers)+'_trainsize_70_trained_on_local_features_'+feature_layer+'_'+solver+'.sav'
    clf = None
    x_test = None
    y_test = None

    if force_replace or not os.path.isfile(filename) or not os.path.isfile("data_test_sampled/y_"+feature_layer+".txt"):
        if not server_path:
            output_features = '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_list_prefix.txt'
        else:
            output_features = '/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/output_list_prefix_train01.txt'

        #output_features = '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_train_list_prefix.txt'
        #features_directory = '/media/kasparov092/64C6CF47C6CF1866/ubuntu-c3d-examples-outputtraining/'
        if not server_path:
            features_directory = '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/'
        else:
            features_directory = '/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/'

        #output/c3d/v_ApplyEyeMakeup_g01_c05/000001
        if not server_path:
            class_index_path = '/home/kasparov092/sources/c3d/v1.0/data/ucf101/ucfTrainTestlist/classInd.txt'
        else:
            class_index_path = '/root/repos/c3d_on_ucf101/ucfTrainTestlist/classInd.txt'

        class_index = {}

        with open(class_index_path) as f:
           for line in f:
               ic = line.split()
               class_index[ic[1]] = int(ic[0])

        x, y = get_training_matrix(output_features, features_directory, class_index, lst_feature_layer)
        print 'all training data fit into memory, press a key to continue'
        input()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

        print ('saving data to disk... ')

        np.savetxt("data_train_sampled/x_"+feature_layer+".txt", x) # the save is trained on sample of testing
        np.savetxt("data_train_sampled/y_"+feature_layer+".txt", y)

        np.savetxt("data_train_sampled/x_local_train_"+feature_layer+".txt", x_train) # the save is trained on sample of testing
        np.savetxt("data_train_sampled/y_local_train_"+feature_layer+".txt", y_train)
        np.savetxt("data_train_sampled/x_local_test_"+feature_layer+".txt", x_test) # the save is trained on sample of testing
        np.savetxt("data_train_sampled/y_local_test_"+feature_layer+".txt", y_test)

        print 'initializing the mlp classifier...'

        clf = MLPClassifier(solver=solver, alpha=1e-5,
                             hidden_layer_sizes=num_hidden_layers, random_state=1)

        print 'preprocessing: scaling the data train and test ...'

        mean_local = x_train.mean(axis=0)
        std_local = x_train.std(axis=0)

        x_train = (x_train - mean_local)/std_local
        x_test = (x_test - mean_local)/std_local

        print 'fitting the classifier to the data...'

        clf.fit(x_train, y_train)
        pickle.dump(clf, open(filename, 'wb'))
    else:
        # load the model from disk
        x_train = np.loadtxt("data_train_sampled/x_local_train_" + feature_layer + ".txt")
        x_test = np.loadtxt("data_train_sampled/x_local_test_" + feature_layer + ".txt")
        y_test = np.loadtxt("data_train_sampled/y_local_test_" + feature_layer + ".txt")

        print 'preprocessing: scaling the test data ...'
        mean_local = x_train.mean(axis=0)
        std_local = x_train.std(axis=0)

        x_test = (x_test - mean_local)/std_local

        print 'Load the model...'
        clf = pickle.load(open(filename, 'rb'))

    print 'predicting: '

    testing_predictions = clf.predict(x_test)
    np.savetxt("data_test/testing_predictions_"+feature_layer+".txt", testing_predictions)
    utl.calculate_accuracy(testing_predictions, y_test)

make_experiments(50, ['pool5'], 'adam', True, True)
exit()
make_experiments(50, ['fc8'], 'adam', True)
make_experiments(50, ['fc6'], 'lbfgs', True)
make_experiments(50, ['fc8'], 'lbfgs', True)

make_experiments(50, ['fc6','fc8'], 'adam', True)
make_experiments(50, ['fc8','fc6'], 'adam', True)
make_experiments(50, ['fc8','fc6'], 'lbfgs', True)
make_experiments(50, ['fc6','fc8'], 'lbfgs', True)
exit()
make_experiments(101, ['fc6'], 'adam', True)
make_experiments(101, ['fc8'], 'adam', True)
make_experiments(101, ['fc6'], 'lbfgs', True)
make_experiments(101, ['fc8'], 'lbfgs', True)
make_experiments(101, ['fc8','fc6'], 'adam', True)
make_experiments(101, ['fc6','fc8'], 'adam', True)
make_experiments(101, ['fc8','fc6'], 'lbfgs', True)
make_experiments(101, ['fc6','fc8'], 'lbfgs', True)

make_experiments(200, ['fc6'], 'adam', True)
make_experiments(200, ['fc8'], 'adam', True)
make_experiments(200, ['fc6'], 'lbfgs', True)
make_experiments(200, ['fc8'], 'lbfgs', True)
make_experiments(200, ['fc8','fc6'], 'adam', True)
make_experiments(200, ['fc6','fc8'], 'adam', True)
make_experiments(200, ['fc8','fc6'], 'lbfgs', True)
make_experiments(200, ['fc6','fc8'], 'lbfgs', True)

make_experiments(3000, ['fc6'], 'adam', True)
make_experiments(3000, ['fc8'], 'adam', True)
make_experiments(3000, ['fc6'], 'lbfgs', True)
make_experiments(3000, ['fc8'], 'lbfgs', True)
make_experiments(3000, ['fc8','fc6'], 'adam', True)
make_experiments(3000, ['fc6','fc8'], 'adam', True)
make_experiments(3000, ['fc8','fc6'], 'lbfgs', True)
make_experiments(3000, ['fc6','fc8'], 'lbfgs', True)

make_experiments(4096, ['fc6'], 'adam', True)
make_experiments(4096, ['fc8'], 'adam', True)
make_experiments(4096, ['fc6'], 'lbfgs', True)
make_experiments(4096, ['fc8'], 'lbfgs', True)
make_experiments(4096, ['fc8','fc6'], 'adam', True)
make_experiments(4096, ['fc6','fc8'], 'adam', True)
make_experiments(4096, ['fc8','fc6'], 'lbfgs', True)
make_experiments(4096, ['fc6','fc8'], 'lbfgs', True)

make_experiments(5000, ['fc6'], 'adam', True)
make_experiments(5000, ['fc8'], 'adam', True)
make_experiments(5000, ['fc6'], 'lbfgs', True)
make_experiments(5000, ['fc8'], 'lbfgs', True)
make_experiments(5000, ['fc8','fc6'], 'adam', True)
make_experiments(5000, ['fc6','fc8'], 'adam', True)
make_experiments(5000, ['fc8','fc6'], 'lbfgs', True)
make_experiments(5000, ['fc6','fc8'], 'lbfgs', True)