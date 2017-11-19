import numpy as np
import array

def display_feature(feature_file):
    with  open(feature_file, "rb") as f:
        line = f.read()

    (n, c, l, h, w) = array.array("i", line[:20])
    feature_vec = np.array(array.array("f", line[20:]))

    print  (n, c, l, h, w)
    return feature_vec

if __name__ == '__main__':
    print 'pool5'
    pool5 = display_feature('/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/output/c3d/v_YoYo_g25_c05/000001.pool5')
    print pool5[0,0,0,...]
    print 'fc6'
    display_feature('/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/output/c3d/v_YoYo_g25_c05/000001.fc6')
    print 'fc8'
    display_feature('/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/output/c3d/v_YoYo_g25_c05/000001.fc8')