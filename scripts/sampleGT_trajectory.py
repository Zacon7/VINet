#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sampling VICON ground truth camera pose. (absolute pose)

In EuRoC MAV dataset, the vicon motion capture system (Leica MS50) record 
data with 100Hz.  (All pose in vicon seems to be global pose, which is
the pose related to first camera pose.)

Because VINet prediction trajectory  with the frequency equal to image 
frame rate, the "answer" of the training need to be in the same frequency.

My quick workaround is to find the nearest timestamp in vicon/data.csv based 
on the timestamp of cam0/.

"""


from PIL import Image
import os
import sys
import errno
from subprocess import call
import csv



def getMidiumIndex(startTime, endTime, imu_index):
    start_index = 0
    end_index = 0
    for i in range(len(imu_index)):
        if imu_index[i] >= startTime:
            start_index = i
            break
            
    for i in range(len(imu_index)):            
        if imu_index[i] >= endTime:
            end_index = i
            break
     
    return int((end_index - start_index) / 2) + start_index
        
def getClosestIndex(searchTime, searchStartIndex, timeList):
    found_idx = -1
    for i in range(searchStartIndex+1, len(timeList)):
        if timeList[i] >= searchTime:
            found_idx = i
            break
    return found_idx

def _get_filenames_and_classes(dataset_dir):

    ## Get image list
    img_list = os.listdir(dataset_dir + '/cam0/data')
    # img_list = os.listdir(dataset_dir + '/cam1/data')
    img_list.sort()

    for i in range(len(img_list)):
        img_list[i] = img_list[i][0:-4]
        # img_list[i] = img_list[i][0:-13]
    
    
    ## Get Pose original data
    pose_datarows = []
    file_path = dataset_dir + '/vicon0/data.csv'
    # file_path = dataset_dir + '/reference/data.csv'
    print("\nWrite to file:", file_path)
    with open(file_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            pose_datarows.append(row)
    pose_datarows = pose_datarows[1:]
    print("Number of rows read:", len(pose_datarows))        

    ## Get Pose time stamps
    time_stamps = []
    for i in range(len(pose_datarows)):
        time_stamps.append(int(pose_datarows[i][0]))
    
    sampled_rows = []
    search_index = 0
    for i in range(len(img_list)):
        search_time = int(img_list[i])
        search_index = getClosestIndex(search_time, search_index, time_stamps)
        sampled_rows.append(pose_datarows[search_index])

    sampled_path = dataset_dir + '/vicon0/sampled.csv'
    # sampled_path = dataset_dir + '/reference/sampled.csv'
    print("Write to file:", sampled_path)
    print("Number of rows written:", len(img_list))
    with open(sampled_path, 'w+') as f:
        for i in range(len(sampled_rows)):
            tmp_str = ",".join(sampled_rows[i])
            f.write(tmp_str + '\n')
    f.close()
        
    
    return
                

def main():
    # _get_filenames_and_classes('data/MH_01_easy/mav0')
    # _get_filenames_and_classes('data/MH_02_easy')
    # _get_filenames_and_classes('data/MH_03_medium')
    # _get_filenames_and_classes('data/MH_04_difficult')
    # _get_filenames_and_classes('data/MH_05_difficult')

    _get_filenames_and_classes('data/V1_01_easy/mav0')
    # _get_filenames_and_classes('data/V1_02_medium/mav0')
    # _get_filenames_and_classes('data/V1_03_difficult/mav0')
    # _get_filenames_and_classes('data/V2_01_easy/mav0')
    # _get_filenames_and_classes('data/V2_02_medium/mav0')
    # _get_filenames_and_classes('data/V2_03_difficult/mav0')
    # _get_filenames_and_classes('data/hy-data')
       
if __name__ == "__main__":
    main()