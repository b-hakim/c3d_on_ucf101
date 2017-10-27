import os

def find_name_issues(dataset_path, labels_dir):
    files=['testlist01.txt','testlist02.txt','testlist03.txt',
           'trainlist01.txt','trainlist02.txt','trainlist02.txt']

    for fname in files:
        filepath = os.path.join(labels_dir, fname)
        with open(filepath) as f:
            for l in f:
                if not os.path.exists(os.path.join(dataset_path, l.split()[0])):
                    print l, "in", filepath

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="the dir containing the directory")
    parser.add_argument("train_test_path", help="the dir containing the train/test labels")
    try:
        args = parser.parse_args()
    except:
        pass
        #args = parser.parse_args(['',''])

    find_name_issues(args.dataset_dir, args.train_test_path)
    print "Done!"
