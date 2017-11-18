import numpy as np

def display_feature(feature_file):
    with  open(feature_file, "rb") as f:
        line = f.read()

    (n, c, l, h, w) = np.array("i", line[:20])
    feature_vec = np.array("f", line[20:])

    print  (n, c, l, h, w)
