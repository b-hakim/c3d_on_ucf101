import Util as utl 

def validate_dataset(dataset_pathes, video_clip_size, output_file_path):
    lines = []
    with open(dataset_pathes) as file:
        lines = file.readlines()
    old_clip = ''
    old_path = ''
    count_removed = 0
    all_files=[]

    for line in lines:
        all_files += [line]

        if old_clip != '':
            clip_name = utl.get_file_name_from_path_without_extention(line.split()[0]+'.')

            if old_clip != clip_name or line == lines[-1]:
                if line == lines[-1]:
                    old_path=line

                index = int(old_path.split()[1])
                index_to_check = index + video_clip_size

                if not utl.is_file_exists(old_path.split()[0] + '/{:06}'.format(index_to_check) + '.jpg'):
                    all_files.remove(old_path)
                    print old_path

                old_clip = clip_name

        else: #occurs only first time
            old_clip = utl.get_file_name_from_path_without_extention(line.split()[0]+'.')
            old_path = line

        old_path = line

    with open(output_file_path, 'w') as file_writer:
        file_writer.writelines(all_files)

    print 'Validation terminated'

#validate_dataset('/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning_trial/test_01.categorized.lst', 16,
#                 '/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning_trial/test_01.categorized.validated.lst')
validate_dataset('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/train_01.validated2.lst', 16,
                 '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/train_01.validated.lst')
