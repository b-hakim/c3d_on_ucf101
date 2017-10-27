def get_clips_from_file_not_in_file(main_file, sub_file, result_file):
    dup_files = []

    with open(sub_file) as file:
        for line in file:
            dup_files += [line]

    ret_files = []

    with open(main_file) as file:
        for line in file:
            if dup_files.__contains__(line):
                continue
            ret_files += [line]

    with open(result_file, 'w') as fw:
        for l in ret_files:
            fw.write(l)

get_clips_from_file_not_in_file('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/test_01.bk.lst',
                                    '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/test_01.categorized.validated.txt',
                            '/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/test_01.full_minus_categorized.bk.lst')