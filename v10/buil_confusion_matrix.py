from pydoc import _OldStyleClass

import Util as utl
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools


def get_file_path_and_action_class(list_lines):
    ret_list = []

    for line in list_lines:
        line_splitted = line.split()
        relative_file_folder = utl.get_file_name_from_path_without_extention(line_splitted[0] + '.')
        file_name = "/{:06}.jpg".format(int(line_splitted[1]))
        ret_list += [[relative_file_folder + file_name, int(line_splitted[2])]]

    return ret_list


# #########################################################################3

def save_matrix_in_file(file_path, conf_mat):
    with open(file_path, 'w') as fw:
        for i in range(len(conf_mat)):
            for j in range(len(conf_mat[i])):
                fw.write(str(conf_mat[i][j]) + ' ')
            fw.write('\n')

def load_matrix_from_file(file_path):
    confusion_matrix = []

    with open(file_path) as fr:
        confusion_matrix = map(lambda x: map(lambda y: float(y), x.split()), fr.readlines())

    return np.array(confusion_matrix)
def load_labels(labels_path):
    labels={}

    with open(labels_path) as labels_file:
        for line in labels_file.readlines():
            labels[float(line.split()[0]) - 1] = line.split()[1]
            labels[line.split()[1]] = float(line.split()[0]) - 1

    return labels

def build_plot_confusion_matrix(input_test_file_path, probabilities_path, classes):#, reset_conf_mat = False
    block_class = []
    dataset_classes = {}
    confusion_matrix = np.zeros([101, 101])

    #if reset_conf_mat:
    #    os.remove("conf_mat.txt");

    if os.path.isfile("conf_mat.txt"):
        confusion_matrix = load_matrix_from_file("conf_mat.txt")
    else:
        with open(classes) as classes_file:
            for line in classes_file:
                class_name = line.split()[1]
                if class_name == 'HandStandPushups':
                    class_name = 'HandstandPushups'
                dataset_classes[int(line.split()[0]) - 1] = class_name
                dataset_classes[class_name] = int(line.split()[0]) - 1

        with open(input_test_file_path) as test_file:
            block_class = get_file_path_and_action_class(test_file.readlines())

        directories_subfolders_files = [x for x in os.walk(probabilities_path)]
        # loop on all clips
        #   set c as current clip index
        #   loop on all blocks
        #       set p as the predicted class index for the current block
        #       increment the confusion_matrix[c, p] by one
        # #
        directories_subfolders_files.remove(directories_subfolders_files[0])

        for clipdir_subfolders_probs in sorted(directories_subfolders_files):
            class_name = utl.get_file_name_from_path_without_extention(clipdir_subfolders_probs[0] + '.')[2:-8]

            if class_name == 'HandStandPushups':
                class_name = 'HandstandPushups'

            class_index = dataset_classes[class_name]

            blocks_prob = sorted(filter(lambda x: x if x.__contains__('prob2') else None,
                                        clipdir_subfolders_probs[2]))

            for prob in blocks_prob:
                predicted_probs = []
                with open(clipdir_subfolders_probs[0] + '/' + prob) as prob_file:
                    predicted_probs = prob_file.readline().split(',')[5:]
                predicted_class_index = predicted_probs.index(max(predicted_probs))
                confusion_matrix[class_index, predicted_class_index] += 1
        save_matrix_in_file("conf_mat.txt", confusion_matrix)

    plot_confusion_matrix(confusion_matrix)


# #########################################################################3
def plot_confusion_matrix(confmat):

    ticks = np.linspace(0, 100, num=101)
    plt.imshow(confmat, interpolation='none')
    plt.colorbar()
    plt.xticks(ticks, fontsize=6)
    plt.yticks(ticks, fontsize=6)
    plt.grid(False)
    plt.show()


def plot_confusion_matrix_old2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix_old1(conf_arr):
    # conf_arr = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
    #            [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #            [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
    #            [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
    #            [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
    #            [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
    #            [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
    #            [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
    #            [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
    #            [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
    #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabet = range(101)
    # plt.xticks(range(width), alphabet[:width])
    # plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')


build_plot_confusion_matrix(
    '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/test_01.categorized.validated.txt',
    '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/output/c3d/',
    '/home/kasparov092/sources/c3d/v1.0/data/ucf101/ucfTrainTestlist/classInd.txt')
