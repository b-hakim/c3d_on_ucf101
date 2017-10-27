import numpy as np
import buil_confusion_matrix as cf

# given a confusion matrix, generate the following statistics
# precision and recall for each class
#
# average precision and average recall
# average accuracy
# error rate
# F-score

def save_precision_recall_each_class(conf_mat, labels, l):
    classname_precision_recall = []

    for i in range(l):
        precision_class_i = conf_mat[i, i] / sum(conf_mat[:, i])
        recall_class_i = conf_mat[i, i] / sum(conf_mat[i, :])
        classname_precision_recall += [labels[i], precision_class_i, recall_class_i]

    i = 0
    with open('precision_recall.txt', 'w') as fr:
        fr.write("{:20}".format("Class Name") + "{:20}".format(" Precision")
                 + "{:20}".format("  Recall") + "{:20}".format("  F1-Score"))
        fr.write('\n')
        p = 0
        r = 0
        for a in classname_precision_recall:
            fr.write('{:20}'.format(str(a)) + ' ')
            if i == 1:
                p = a
            if i == 2:
                r = a
                fr.write(str(float(fscore(1, p, r))) + '\n')
                i = -1
            i += 1

def save_micro_macro_precision_recall(conf_mat, num_classes):
    macro_precision = 0
    macro_recall = 0

    for i in range(num_classes):
        macro_precision += conf_mat[i, i] / sum(conf_mat[:, i])
        macro_recall += conf_mat[i, i] / sum(conf_mat[i, :])

    macro_precision = macro_precision / float(num_classes)
    macro_recall = macro_recall / float(num_classes)

    m = np.array(conf_mat)
    micro_precision_recall = sum(m.diagonal()) / float(m.sum())

    f1score_macro = fscore(1, macro_precision, macro_recall)
    f1score_micro = fscore(1, micro_precision_recall, micro_precision_recall)

    with open('micro_macro_precision_recall.txt', 'w') as fr:
        fr.write('macro precision: ' + str(macro_precision))
        fr.write('\nmacro recall: ' + str(macro_recall))
        fr.write('\nmicro precision: ' + str(micro_precision_recall))
        fr.write('\nmicro recall: ' + str(micro_precision_recall))
        fr.write('\nmacro f1-score: ' + str(f1score_macro))
        fr.write('\nmicro f1-score: ' + str(f1score_micro))

def save_avg_accuracy_avg_error(conf_mat, num_classes):
    # accuracy:
    correctly_classified = conf_mat.diagonal().sum()
    all_classifications = conf_mat.sum()
    accuracy = float(correctly_classified) / all_classifications
    error = 1 - accuracy
    avg_error = 0
    avg_accuracy = 0

    for i in range(num_classes):
        tp = conf_mat[i, i]
        tn = conf_mat[0:i, 0:i].sum() + conf_mat[0:i, i + 1:num_classes].sum() + \
             conf_mat[i + 1:num_classes, 0:i].sum() + conf_mat[i + 1:num_classes, i + 1:num_classes].sum()
        fp = conf_mat[:, i].sum() - conf_mat[i, i]
        fn = conf_mat[i, :].sum() - conf_mat[i, i]

        avg_accuracy += float(tp + tn) / float(fp + fn + tp + tn)
        avg_error += float(fp + fn) / float(fp + fn + tp + tn)

    avg_accuracy /= float(num_classes)
    avg_error /= float(num_classes)

    with open('accuracy_error.txt', 'w') as fr:
        fr.write("Accuracy: " + str(float(accuracy)))
        fr.write("\nError: " + str(float(error)))
        fr.write("\nAvg. Accuracy: " + str(float(avg_accuracy)))
        fr.write("\nAvg. Error: " + str(float(avg_error)))

def fscore(beta, precision, recall):
    return (1 + beta ** 2) * (float(precision * recall) / ((beta ** 2 * precision) + recall))

def get_precision_recall_fscore_lower_than(val=0.6):
    lines = []

    with open("precision_recall.txt") as file:
        lines = map(lambda x: x.split(), file.readlines())

    lines.remove(lines[0])

    with open("classes_lessthan_60_precision.txt", 'w') as file:
        for r in lines:
            if float(r[1]) < val:
                file.write(r[0] + ' : ' + r[1] + ' ' + r[2] + ' ' + r[3] + '\n')

    with open("classes_lessthan_60_recall.txt", 'w') as file:
        for r in lines:
            if float(r[2]) < val:
                file.write(r[0] + ' : ' + r[1] + ' ' + r[2] + ' ' + r[3] + '\n')

    with open("classes_lessthan_60_f1-score.txt", 'w') as file:
        for r in lines:
            if float(r[3]) < val:
                file.write(r[0] + ' : ' + r[1] + ' ' + r[2] + ' ' + r[3] + '\n')

def save_confusions_labels_for_low_f1_score(conf_mat, labels, fscore_file_path):
    fscore_listing = []

    with open(fscore_file_path) as fr:
        fscore_listing = fr.readlines()

    with open('confused_matrix_low_f1_score.txt', 'w') as file:
        for _class in fscore_listing:
            class_name = _class.split()[0]
            index = int(labels[class_name])
            confused_with_classes = np.arange(101)[conf_mat[index] != 0]
            file.write(class_name+": \n")

            for c in confused_with_classes:
                file.write("{:>20}".format(labels[c]) + ' -- ' + "{:4}".format(conf_mat[index, c])+"\n")

            file.write('\n')


def confusion_matrix_stats(confusion_matrix_path, labels_path, num_classes):
    confusion_matrix = []

    #with open(confusion_matrix_path) as fr:
    #    confusion_matrix = map(lambda x: map(lambda y: float(y), x.split()), fr.readlines())
    confusion_matrix = cf.load_matrix_from_file(confusion_matrix_path)
    labels=cf.load_labels(labels_path)

    save_precision_recall_each_class(np.array(confusion_matrix), labels, num_classes)
    save_micro_macro_precision_recall(np.array(confusion_matrix), num_classes)
    save_avg_accuracy_avg_error(np.array(confusion_matrix), num_classes)
    get_precision_recall_fscore_lower_than()
    save_confusions_labels_for_low_f1_score(confusion_matrix, labels, 'classes_lessthan_60_f1-score.txt')

confusion_matrix_stats('/home/kasparov092/PycharmProjects/UCF101_example/conf_mat.txt',
                       '/home/kasparov092/PycharmProjects/UCF101_example/classInd.txt',
                       101)
