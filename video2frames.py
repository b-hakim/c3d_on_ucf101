#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 21:45:02 2017

@author: kasparov092
"""
import SaveFrameFromVideo as sffv
import Util as utl
import os

#def SelectFramesFromVideo(videos_path, output_path):
def SelectFramesFromVideo(dataset_dir, videos_path, output_path):
    last_video_name = ''

    with open(videos_path) as file:
        for line in file:
            line = line.strip()
            video_from_to = line.split()

            video_name = utl.get_file_name_from_path_with_extention(video_from_to[0])

            if video_name == last_video_name:
                continue
            else:
                last_video_name = video_name

            #sffv.select_all_frames_from_video(video_from_to[0].replace('frm', 'video')+'.avi', output_path)
            sffv.select_all_frames_from_video(os.path.join(dataset_dir,video_from_to[0]), output_path)

def ValidateAllFramesInVideoAreExtracted(dataset_dir, videos_path, output_path):
    last_video_name = ''
    lst_vid_issues = []

    with open(videos_path) as file:
        for line in file:
            line = line.strip()
            video_from_to = line.split()

            video_name = utl.get_file_name_from_path_with_extention(video_from_to[0])

            if video_name == last_video_name:
                continue
            else:
                last_video_name = video_name

            #sffv.select_all_frames_from_video(video_from_to[0].replace('frm', 'video')+'.avi', output_path)
            video_path = os.path.join(dataset_dir,video_from_to[0])

            length = utl.get_num_frames(video_path)
            video_name = utl.get_file_name_from_path_without_extention(video_path)
            video_category = utl.get_direct_folder_containing_file(video_path)
            save_frame_path = os.path.join(output_path, video_category, video_name)
            save_frame_full_path = save_frame_path + '/{:06}'.format(length) + ".jpg"

            if not os.path.exists(save_frame_full_path):
                lst_vid_issues += [video_from_to[0]]

    print lst_vid_issues
    return lst_vid_issues

def GetFramesFromVideosIn(dataset_dir, output_path, ls):
    for line in ls:
        line = line.strip()
        video_from_to = line
        print "getting frames for:", video_from_to
        sffv.select_all_frames_from_video(os.path.join(dataset_dir,video_from_to), output_path)

def SelectFramesFromVideoButNotIn(videos_path, video_already_exists, output_path):
    lines = []
    last_video_name = ''
    video_already_exists_pathes = []

    with open(videos_path) as file:
        for line in file:
            video_already_exists_pathes += [line]

    print 'pathes = ', len(video_already_exists_pathes)


    with open(videos_path) as file:
        i=0
        for line in file:
            line = line.strip() #or someother preprocessing
            video_from_to = line.split()

            if video_already_exists_pathes.__contains__(video_from_to[0]):
                continue
           # sffv.select_frame_from_video(video_from_to[0][0:len(video_from_to[0])-1]+'.avi',
           #                             int(video_from_to[1]),
           #                               output_path);
            video_name = utl.get_file_name_from_path_with_extention(video_from_to[0])

            if video_name == last_video_name:
                continue
            else:
                last_video_name = video_name

            sffv.select_all_frames_from_video(video_from_to[0].replace('frm', 'video')+'.avi', output_path)

def ParseArgs():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="path to the input videos dir")
    parser.add_argument("label_path", help="path to the train/test file")
    parser.add_argument("output_dir", help="path to the output frames dir")
    return parser.parse_args()

if __name__ == '__main__':
    args = ParseArgs()
    #SelectFramesFromVideoButNotIn('/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning/test_01.lst',
    #                  '/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning/train_01.lst',
    #                  '/home/kasparov092/sources/c3d/v1.0/data/ucf101/frm/')
    #SelectFramesFromVideo(args.dataset_dir, args.label_path, args.output_dir)
    ls = ValidateAllFramesInVideoAreExtracted(args.dataset_dir, args.label_path, args.output_dir)
    GetFramesFromVideosIn(args.dataset_dir, args.output_dir, ls)

    #'/home/kasparov092/sources/c3d/v1.0/data/ucf101/frm/')


    #SelectFramesFromVideo('/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning/train_01.lst',
    #                      '/home/kasparov092/sources/c3d/v1.0/data/ucf101/frm/')

    #sffv.select_frame_from_video('/home/kasparov092/sources/c3d/data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',
    #                        1,
    #                        '/home/kasparov092/Desktop/UCF101_Frames/')