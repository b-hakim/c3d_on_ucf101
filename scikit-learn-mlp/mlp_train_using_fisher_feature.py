from sklearn.neural_network import MLPClassifier
import Utility as utl
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import fisher_encoding
from sklearn.mixture import GMM

def one_hot_encoding_class(class_name, class_index):
    i = class_index[class_name]
    out = np.zeros(101)
    out[i-1] = 1
    return out


def get_global_FISHER_features_training_matrix(output_features, features_directory, class_index, features_used=['.fc6.txt']):
    lines = []
    ret_x = []
    ret_y = []

    with open(output_features) as f:
        lines = f.readlines()

    old_directory_name = ''
    same_clip_features = []

    for line in lines:
        line = features_directory+line.rstrip()
        file_dir = utl.get_file_directory(line)
        features_vector = []

        for f in features_used:
            with open(line + f) as fil:
                features_vector += map(lambda x: float(x), fil.readline().split(',')[5:])

        if file_dir != old_directory_name:
            if same_clip_features != []:
                #ret_x.append(same_clip_features)
                ret_y.append(one_hot_encoding_class(line[85:line[85:].index('_') + 85],  # 85, nada: 73
                                                    class_index))
                gmm = GMM(n_components=2, covariance_type='diag')
                gmm.fit(same_clip_features)

                fv = fisher_encoding.fisher_vector(same_clip_features, gmm)
                ret_x.append(fv)
            old_directory_name = file_dir
            same_clip_features = [np.array(features_vector)]
        else:
            same_clip_features.append(features_vector)

    gmm = GMM(n_components=1, covariance_type='diag')
    gmm.fit(same_clip_features)

    fv = fisher_encoding.fisher_vector(same_clip_features, gmm)
    ret_x.append(fv)
    ret_y.append(one_hot_encoding_class(line[85:line[85:].index('_') + 85],  # 85, nada: 73
                                        class_index))

    return ret_x, ret_y


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
x, y = get_global_FISHER_features_training_matrix(output_features, features_directory, class_index)


#np.savetxt("data_test_sampled/x_global_FISHER.txt", x) # the save is trained on sample of testing
#np.savetxt("data_test_sampled/y_global_FISHER.txt", y)

x_global_train, x_global_test, y_global_train, y_global_test = train_test_split(x, y, test_size=0.30, random_state=42)

x_global_train = np.array(x_global_train)
x_global_test = np.array(x_global_test)
y_global_train = np.array(y_global_train)
y_global_test = np.array(y_global_test)

#np.savetxt("data_test_sampled/x_global_train_FISHER.txt", x_global_train) # the save is trained on sample of testing
#np.savetxt("data_test_sampled/y_global_train_FISHER.txt", y_global_train)
#np.savetxt("data_test_sampled/x_global_test_FISHER.txt", x_global_test) # the save is trained on sample of testing
#np.savetxt("data_test_sampled/y_global_test_FISHER.txt", y_global_test)
#exit()
#x_global_train = np.loadtxt("data_test_sampled/x_global_train_FISHER.txt")
#y_global_train = np.loadtxt("data_test_sampled/y_global_train_FISHER.txt")
#x_global_test = np.loadtxt("data_test_sampled/x_global_test_FISHER.txt")
#y_global_test = np.loadtxt("data_test_sampled/y_global_test_FISHER.txt")

mean = x_global_train.mean(axis=0)
std = x_global_train.std(axis=0)

x_global_train = (x_global_train-mean)/std
x_global_test = (x_global_test-mean)/std

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=3000, random_state=1)

clf.fit(x_global_train, y_global_train)

filename = 'data_test_sampled/model_3000_trainsize_70_trained_on_global_FISHER_features.sav'
#pickle.dump(clf, open(filename, 'wb'))

testing_predictions = clf.predict(x_global_test)
#np.savetxt('data_test_sampled/testing_predictions_global_FISHER_model_global_FISHER_test.txt', testing_predictions)

print "Using a model trained on global features only, testing with global features:"
utl.calculate_accuracy(testing_predictions, y_global_test)

exit()

#====================================================================================
filename = 'data_test_sampled/model_3000_trainsize_70_trained_on_local_features.sav'
clf2 = pickle.load(open(filename, 'rb'))

print "Using a model trained on local features only, testing with global features:"
testing_predictions = clf2.predict(x_global_test)
utl.calculate_accuracy(testing_predictions, y_global_test)
#====================================================================================
x_local_test = np.loadtxt("data_test_sampled/x_local_test.txt") # the save is trained on sample of testing
y_local_test = np.loadtxt("data_test_sampled/y_local_test.txt")

print "Using a model trained on global features only, testing with local features:"
testing_predictions = clf.predict(x_local_test)
utl.calculate_accuracy(testing_predictions, y_local_test)
#====================================================================================
x_local_test = np.loadtxt("data_test_sampled/x_local_test.txt") # the save is trained on sample of testing
y_local_test = np.loadtxt("data_test_sampled/y_local_test.txt")

print "Using a model trained on local features only, testing with local features:"
testing_predictions = clf2.predict(x_local_test)
utl.calculate_accuracy(testing_predictions, y_local_test)


