#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Convert sampled ground truth(/state_groundtruth_estimate0/sampled.csv) to relative pose

The format is in se3 R6 = (ρx, ρy, ρz, Φ1, Φ2, Φ3)

Relative pose transition is simply calculated by:
    previous position P = (x, y, z)
    current position P'=(x', y', z')
    relative pose transition R = (x' - x, y'-y, z'-z)

Relative rotation is a little bit complicated:
    'Difference' between two quaternions
    Q = Q2*Q1^{-1}.
    (https://stackoverflow.com/questions/1755631/difference-between-two-quaternions)

"""


from sophus.so3 import So3
from sophus.se3 import Se3
from sophus.quaternion import Quaternion as Qua
from sympy import *
import quaternion
import numpy as np
import csv
import decimal
import errno
import os
import sys
from subprocess import call

from PIL import Image
from tqdm import tqdm

# from sophus import *
sys.path.append("/home/zacon/code_projects/VINet")


# from pyquaternion import Quaternion as Qua


# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 15


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, "f")


# xyz quaternion ==> se(3)
def normalize(ww, wx, wy, wz):  # make first number positive
    q = [ww, wx, wy, wz]
    # Find first negative
    idx = -1
    for i in range(len(q)):
        if q[i] < 0:
            idx = i
            break
        elif q[i] > 0:
            break
    # -1 if should not filp, >=0  flipping index
    if idx >= 0:
        ww = ww * -1
        wx = wx * -1
        wy = wy * -1
        wz = wz * -1
    return ww, wx, wy, wz


def xyzQuaternion2se3_(arr):
    x, y, z, ww, wx, wy, wz = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]
    trans = Matrix([x, y, z])
    ww, wx, wy, wz = normalize(ww, wx, wy, wz)

    q_real = ww
    q_img = Matrix([wx, wy, wz])

    q = Qua(q_real, q_img)
    R = So3(q)
    RT = Se3(R, trans)

    numpy_vec = np.array(RT.log()).astype(float)  # SE3 to se3 KAATUU TAHAN!!!!

    return np.concatenate(numpy_vec)


def _get_filenames_and_classes(dataset_dir):
    relative_pose = []  # relative camera pose
    file_path = dataset_dir + "/state_groundtruth_estimate0/sampled_relative.csv"
    # file_path = dataset_dir + '/reference/sampled_relative.csv'

    print("\nRead data from file:", file_path)
    with open(file_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in spamreader:
            relative_pose.append(row)
    print("Number of rows read:", str(len(relative_pose)))

    # Calculate relative pose
    trajectory_relative = []
    for i in tqdm(range(len(relative_pose))):
        # timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        se3R6 = xyzQuaternion2se3_(
            [
                float(relative_pose[i][1]),
                float(relative_pose[i][2]),
                float(relative_pose[i][3]),
                float(relative_pose[i][4]),
                float(relative_pose[i][5]),
                float(relative_pose[i][6]),
                float(relative_pose[i][7]),
            ]
        )
        trajectory_relative.append(se3R6)
        # print(i)

    file_path = dataset_dir + "/state_groundtruth_estimate0/sampled_relative_R6.csv"
    print("Write to file:", file_path)
    with open(file_path, "w+") as f:
        for i in range(len(trajectory_relative)):
            r1 = float_to_str(trajectory_relative[i][0])
            r2 = float_to_str(trajectory_relative[i][1])
            r3 = float_to_str(trajectory_relative[i][2])
            r4 = float_to_str(trajectory_relative[i][3])
            r5 = float_to_str(trajectory_relative[i][4])
            r6 = float_to_str(trajectory_relative[i][5])
            line_str = (
                str(relative_pose[i][0]) + "," + r1 + "," + r2 + "," + r3 + "," + r4 + "," + r5 + "," + r6
            )
            f.write(line_str + "\n")
    print("Number of rows written:", len(trajectory_relative))
    f.close()
    return


def main():
    print("\nStart process V1_01_easy")
    _get_filenames_and_classes("data/V1_01_easy/mav0")

    print("\nStart process V1_02_medium")
    _get_filenames_and_classes("data/V1_02_medium/mav0")

    print("\nStart process V1_03_difficult")
    _get_filenames_and_classes("data/V1_03_difficult/mav0")

    print("\nStart process V2_01_easy")
    _get_filenames_and_classes("data/V2_01_easy/mav0")

    print("\nStart process V2_02_medium")
    _get_filenames_and_classes("data/V2_02_medium/mav0")

    print("\nStart process V2_03_difficult")
    _get_filenames_and_classes("data/V2_03_difficult/mav0")


if __name__ == "__main__":
    main()
