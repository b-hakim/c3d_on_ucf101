from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
from sklearn import preprocessing


def one_hot_encoding_class(class_name, class_index):
   i = class_index[class_name]
   out = np.zeros(101)
   out[i-1] = 1
   return out

def get_training_matrix(output_features, features_directory, class_index, features_used=['.fc6.txt', '.fc8.txt']):
   lines = []
   with open(output_features) as f:
       lines = f.readlines()

   features_file_pathes_used = {}

   for line in lines:
       full_line = features_directory+line.rstrip()
       features_file_pathes_used[full_line] = []

       for f in features_used:
           features_file_pathes_used[full_line] += [f[6:]]

   ret_x = []
   ret_y = []

   for feature_set_key in features_file_pathes_used.keys():
       sub_features = []

       for sub_feature in features_file_pathes_used[feature_set_key]:
           with open(feature_set_key + sub_feature) as fil:
               sub_features += map(lambda x: float(x), fil.readline().split(',')[5:])


       ret_x.append(sub_features)
       ret_y.append(one_hot_encoding_class(feature_set_key[73:feature_set_key[73:].index('_')+73],
                                           class_index))

   return ret_x, ret_y


output_features = '/home/nada/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_list_prefix.txt'
features_directory = '/home/nada/sources/c3d/v1.0/examples/c3d_feature_extraction/'
#output/c3d/v_ApplyEyeMakeup_g01_c05/000001
class_index_path = '/home/nada/sources/c3d/v1.0/data/UCF-101/ucfTrainTestlist/classInd.txt'
class_index = {}

with open(class_index_path) as f:
  for line in f:
      ic = line.split()
      class_index[ic[1]] = int(ic[0])

print 'loading sample input/output...'
#
x, y = get_training_matrix(output_features, features_directory, class_index)
#
##np.savetxt("x.txt", x)
##np.savetxt("y.txt", y)
#x = np.loadtxt("x.txt")
#y = np.loadtxt("y.txt")
#
#print 'initializing the mlp classifier...'
#
clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=5000, random_state=1, batch_size='auto')
#
print 'preprocessing: scaling the data ...'
#
#x = preprocessing.scale(x)
#
#print 'fitting the classifier to the data...'
#
#clf.fit(x, y)
#
#filename = 'model_3000.sav'
#pickle.dump(clf, open(filename, 'wb'))
#
## load the model from disk
##loaded_model = pickle.load(open(filename, 'rb'))
#
print 'predicting: '
#
#testing_predictions = clf.predict(x)
testing_predictions = clf.predict(x)
#
np.savetxt("testing_predictions.txt", testing_predictions)
#testing_predictions = np.loadtxt("x.txt")

print 'Calculating accuracies: \n'
correct = 0
incorrect = 0

for i in range (0, len(testing_predictions)):
    if (np.array(testing_predictions[i] == y[i])).all():
        correct += 1
    else:
        incorrect += 1

print "  correct: ", correct, "\nincorrect: " , incorrect