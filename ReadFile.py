#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 21:45:02 2017

@author: kasparov092
"""
import SaveFrameFromVideo as sffv
import Util as utl

#def SelectFramesFromVideo(videos_path, output_path):
def SelectFramesFromVideo(dataset_dir, videos_path, output_path):
    lines = []
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
    SelectFramesFromVideo(args.dataset_dir, args.label_path, args.outputdir)

    #'/home/kasparov092/sources/c3d/v1.0/data/ucf101/frm/')


    #SelectFramesFromVideo('/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning/train_01.lst',
    #                      '/home/kasparov092/sources/c3d/v1.0/data/ucf101/frm/')

    #sffv.select_frame_from_video('/home/kasparov092/sources/c3d/data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',
    #                        1,
    #                        '/home/kasparov092/Desktop/UCF101_Frames/')