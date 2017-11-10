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

def GetFramesFromVideosIn(dataset_dir, videos_path, output_path):
    ls =  ['Basketball/v_Basketball_g16_c01.avi', 'Basketball/v_Basketball_g16_c02.avi',
     'Basketball/v_Basketball_g16_c03.avi', 'Basketball/v_Basketball_g16_c04.avi',
     'Basketball/v_Basketball_g16_c05.avi', 'Basketball/v_Basketball_g16_c06.avi',
     'Basketball/v_Basketball_g17_c02.avi', 'Basketball/v_Basketball_g18_c03.avi',
     'Basketball/v_Basketball_g18_c04.avi', 'Basketball/v_Basketball_g18_c05.avi', 'Biking/v_Biking_g20_c01.avi',
     'Biking/v_Biking_g20_c07.avi', 'Billiards/v_Billiards_g17_c01.avi', 'GolfSwing/v_GolfSwing_g19_c06.avi',
     'GolfSwing/v_GolfSwing_g21_c05.avi', 'GolfSwing/v_GolfSwing_g21_c06.avi', 'HorseRace/v_HorseRace_g20_c01.avi',
     'HorseRiding/v_HorseRiding_g15_c01.avi', 'HorseRiding/v_HorseRiding_g16_c05.avi',
     'HorseRiding/v_HorseRiding_g16_c06.avi', 'JavelinThrow/v_JavelinThrow_g19_c01.avi',
     'JugglingBalls/v_JugglingBalls_g18_c01.avi', 'JugglingBalls/v_JugglingBalls_g18_c02.avi',
     'JugglingBalls/v_JugglingBalls_g18_c03.avi', 'JugglingBalls/v_JugglingBalls_g18_c04.avi',
     'JumpingJack/v_JumpingJack_g20_c01.avi', 'JumpingJack/v_JumpingJack_g20_c03.avi',
     'JumpingJack/v_JumpingJack_g20_c04.avi', 'JumpRope/v_JumpRope_g17_c01.avi', 'JumpRope/v_JumpRope_g17_c02.avi',
     'JumpRope/v_JumpRope_g17_c03.avi', 'JumpRope/v_JumpRope_g17_c04.avi', 'PlayingGuitar/v_PlayingGuitar_g19_c05.avi',
     'PlayingTabla/v_PlayingTabla_g15_c01.avi', 'PlayingTabla/v_PlayingTabla_g15_c03.avi',
     'PlayingTabla/v_PlayingTabla_g15_c04.avi', 'PullUps/v_PullUps_g16_c03.avi', 'PullUps/v_PullUps_g20_c04.avi',
     'SoccerJuggling/v_SoccerJuggling_g16_c01.avi', 'SoccerJuggling/v_SoccerJuggling_g16_c02.avi',
     'SoccerJuggling/v_SoccerJuggling_g16_c03.avi', 'SoccerJuggling/v_SoccerJuggling_g16_c04.avi',
     'SoccerJuggling/v_SoccerJuggling_g16_c07.avi', 'SoccerJuggling/v_SoccerJuggling_g20_c05.avi',
     'SoccerJuggling/v_SoccerJuggling_g21_c01.avi', 'SoccerJuggling/v_SoccerJuggling_g21_c02.avi',
     'SoccerJuggling/v_SoccerJuggling_g21_c03.avi', 'SoccerJuggling/v_SoccerJuggling_g21_c04.avi',
     'TennisSwing/v_TennisSwing_g16_c01.avi', 'TennisSwing/v_TennisSwing_g16_c02.avi',
     'TennisSwing/v_TennisSwing_g16_c03.avi', 'TennisSwing/v_TennisSwing_g16_c04.avi',
     'TennisSwing/v_TennisSwing_g16_c05.avi', 'TennisSwing/v_TennisSwing_g16_c06.avi',
     'TennisSwing/v_TennisSwing_g18_c05.avi', 'TennisSwing/v_TennisSwing_g19_c01.avi',
     'TennisSwing/v_TennisSwing_g20_c02.avi', 'TrampolineJumping/v_TrampolineJumping_g15_c05.avi',
     'WalkingWithDog/v_WalkingWithDog_g21_c03.avi']

    last_video_name = ''

    for line in ls:
        line = line.strip()
        video_from_to = line
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
    #ValidateAllFramesInVideoAreExtracted(args.dataset_dir, args.label_path, args.output_dir)
    GetFramesFromVideosIn(args.dataset_dir, args.label_path, args.output_dir)

    #'/home/kasparov092/sources/c3d/v1.0/data/ucf101/frm/')


    #SelectFramesFromVideo('/home/kasparov092/sources/c3d/v1.0/examples/c3d_finetuning/train_01.lst',
    #                      '/home/kasparov092/sources/c3d/v1.0/data/ucf101/frm/')

    #sffv.select_frame_from_video('/home/kasparov092/sources/c3d/data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',
    #                        1,
    #                        '/home/kasparov092/Desktop/UCF101_Frames/')