import numpy as np

def display_feature(feature_file):
    with  open(feature_file, "rb") as f:
        line = f.read()

    (n, c, l, h, w) = np.array("i", line[:20])
    feature_vec = np.array("f", line[20:])

    print  (n, c, l, h, w)

if __name__ == '__main__':
    display_feature('/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/output/c3d/v_ApplyEyeMakeup_g01_c01/')