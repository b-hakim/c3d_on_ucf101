import Utility as utl


def count_clips_in_dataset(output_file_path):
    dic = {}
    total_clips = 0

    with open(output_file_path) as fl:
        for l in fl:
            name = utl.get_file_directory(l)
            if name in dic:
                dic[name] += 1
            else:
                dic[name]  = 1
                total_clips += 1
    return total_clips

print count_clips_in_dataset('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_list_prefix.txt')


