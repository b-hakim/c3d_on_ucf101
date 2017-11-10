import os

def get_file_name_from_path_without_extention(file_path):
    return file_path[file_path.rfind('/') + 1:file_path.rfind('.')]


def get_file_name_from_path_with_extention(file_path):
    return file_path[file_path.rfind('/') + 1:len(file_path)]


def get_direct_folder_containing_file(file_path):
    ind_end = file_path.rfind('/')
    ind_start = file_path[0:ind_end].rfind('/') + 1
    if ind_end == -1:
        raise Exception("Path does not contain folder")
    # print ind_start, ind_end

    return file_path[ind_start:ind_end]


def is_same_group(file1_path, file2_path):
    return file1_path[0:-2] == file2_path[0:-2]


def get_num_frames(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

# test get_file_name_from_path_without_extension
# print 'file without extension: ', \
#    get_file_name_from_path_without_extention('Mainfolder/foldername/filename.avi')
# print 'file without extension: ', \
#   get_file_name_from_path_without_extention('Mainfolder/foldername/filename sadas')

# test get_file_name_from_path_with_extension
# print 'file with extension: ', \
#    get_file_name_from_path_with_extention('foldername/filename.avi')

# test get_file_name_from_path_with_extension
# print 'file with extension: ', \
#    get_file_name_from_path_with_extention('filename')

# test get_direct_folder_containing_file
# print 'folder containing file: ', \
#    get_direct_folder_containing_file('Mainfolder/foldername/filename.avi')

# test get_direct_folder_containing_file
# print 'folder containing file: ', \
#   get_direct_folder_containing_file('Mainfolder/filename.avi')

# test get_file_name_from_path_without_extension
# print 'folder containing file: ', \
#   get_direct_folder_containing_file('filename.avi')

# test is_same_group
#print is_same_group("v_ApplyEyeMakeup_g02_c05", "v_ApplyEyeMakeup_g01_c05")
#print is_same_group("v_ApplyEyeMakeup_g01_c05", "v_ApplyEyeMakeup_g01_c05")
#print is_same_group("v_ApplyEyeMakeup_g01_c05", "v_ApplyEyeMakeup_g02_c05")
def is_file_exists(file_path):
    return os.path.exists(file_path)