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
            output_lines += [clip_full_path + ' ' + str(i)+ ' ' + line.split()[1] + '\n']

    with open(output_file_path, 'w') as file_writer:
        file_writer.writelines(output_lines)
