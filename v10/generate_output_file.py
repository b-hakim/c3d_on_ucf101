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




generate_output_file('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/test_01.full_minus_categorized.validated.lst',
                     '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/',
                     'output_test/c3d/',
                     '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_test_minus_sampled_list_prefix.txt')
