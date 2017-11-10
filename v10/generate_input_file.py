import Util as utl
import os

def generate_input_file(ucf101_input_path, base_dir_frm, output_file_path, isTest):

    lines = []

    with open(ucf101_input_path) as file:
        lines = file.readlines()

    output_lines =[]

    for line in lines:

        dir_name = utl.get_direct_folder_containing_file(line.split()[0])
        clip_name = utl.get_file_name_from_path_without_extention(line.split()[0])
        clip_full_path = os.path.join(base_dir_frm, dir_name, clip_name)

        pics = utl.get_all_files_with_extenstion(clip_full_path, '.jpg')
        n = int(len(pics)/16)*16

        for i in range(1, n, 16):
            if isTest:
                output_lines += [clip_full_path + ' ' + str(i)+ ' 0 ' + '\n']
            else:
                output_lines += [clip_full_path + ' ' + str(i)+ ' ' + line.split()[1] + '\n']

    with open(output_file_path, 'w') as file_writer:
        file_writer.writelines(output_lines)

def ParseArgs():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ucf101_input_file_path", help="path to the input file")
    parser.add_argument("base_dir_frm", help="path to the frm dir path")
    parser.add_argument("output_file_path", help="path to the output file")
    parser.add_argument("isTest", help="is the input train or test file")
    return parser.parse_args()

if __name__ == '__main__':
    args = ParseArgs()
    print args.isTest == True
    generate_input_file(args.ucf101_input_file_path, args.base_dir_frm, args.output_file_path, bool(args.isTest))

#python v10/generate_input_file.py "/root/repos/c3d_on_ucf101/ucfTrainTestlist/trainlist01.txt" "/root/sources/C3D/C3D-v1.0/data/ucf101/frm/" "/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/input_list_frm_train01.txt", False
