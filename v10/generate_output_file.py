import Util as utl
import os

def generate_output_file(input_path, output_folder, output_prefix, output_file):

    lines = []

    with open(input_path) as file:
        lines = file.readlines()

    output_lines =[]

    for line in lines:
        clip_name = utl.get_file_name_from_path_without_extention(line.split()[0]+'.')
        output_lines += [output_folder+output_prefix+clip_name+'/{:06}'.format(int(line.split()[1]))+'\n']

        if not os.path.exists(output_folder + output_prefix + clip_name):
            os.mkdir(output_folder + output_prefix + clip_name)


    with open(output_file, 'w') as file_writer:
        file_writer.writelines(output_lines)

#generate_output_file('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/test_01.categorized.validated.txt',
#                     '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/',
#                     'output/c3d/',
#                     '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_list_prefix.txt')

#generate_output_file('/home/kasparov092/sources/c3d/v1.1/examples/c3d_ucf101_feature_extraction/test_01.categorized.validated.list',
#                     '/home/kasparov092/sources/c3d/v1.1/examples/c3d_ucf101_feature_extraction/',
#                     'output/c3d/',
#                     '/home/kasparov092/sources/c3d/v1.1/examples/c3d_ucf101_feature_extraction/output_features.prefix')

#generate_output_file('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/train_01.validated.lst',
#                     '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/',
#                     'output_train/c3d/',
#                     '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_train_list_prefix.txt')

def ParseArgs():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to the input file")
    parser.add_argument("output_folder", help="path to the output dir path")
    parser.add_argument("output_prefix", help="path to the output file")
    parser.add_argument("output_file", help="the file path to produce")
    return parser.parse_args()

if __name__ == '__main__':
    args = ParseArgs()
    generate_output_file(args.input_file,
                     args.output_folder,
                     args.output_prefix,
                     args.output_file)

# python generate_output_file.py "/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/input_list_frm_train01.txt" "/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/" "output/c3d/" "/root/sources/C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/output_list_prefix_train01.txt "
