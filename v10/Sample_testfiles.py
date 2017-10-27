import random
import Util as utl

# *From each of the 4-to-7 clips inside each group of the 7 groups of the 101 classes, I use 1 random clip.
#  So I have a total of 1*7*101= 707

# * For each clip I use all blocks to reach total of (in average there is 11.05 block) 707*11.05= 7812
# blocks to test.

# * To test, we use batches as all of these blocks doesn't fit in the memory. I use a batch of size
# 28 blocks and a total of 279 batches to cover all of the 7812 blocks.

def random_split_from_each_category(file_path, output_file_path, k, use_category):
    lines = []

    with open(file_path) as file:
        lines = file.readlines()

    same_category_lines =[]
    selected_clips = []

    for line in lines:
        if line != lines[-1] and (len(same_category_lines) == 0 or \
            utl.is_same_group(utl.get_file_name_from_path_without_extention(same_category_lines[0].split()[0]+'.'),\
                              utl.get_file_name_from_path_without_extention(line.split()[0]+'.'))):
            same_category_lines += [line]
        else:
            if line == lines[-1]:
                same_category_lines+=[line]

            # find the number of clips in the previous group
            num_clips = int(same_category_lines[-1].split()[0][-2:len(same_category_lines[-1])])
            selected_clip = random.sample(range(1, num_clips + 1), k)

            # select all the blocks of the selected clips
            base_file_name = utl.get_file_name_from_path_without_extention(same_category_lines[0].split()[0]+'.')

            selected_file_names = []

            for i in range(0,len(selected_clip)):
                selected_file_names += [str(base_file_name [0:-2]) + "{:02}".format(selected_clip[i])]

           # print file_names

            for new_line in same_category_lines:
                for selected_file_name in selected_file_names:
                    if selected_file_name in new_line:
                        selected_clips += new_line

            same_category_lines = [line]

    with open(output_file_path, 'w') as file_writer:
        file_writer.writelines(selected_clips)


random_split_from_each_category('/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning_trial/test_01.bk.lst',
                                '/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning_trial/test_01.categorized.lst',
                                1, False)

