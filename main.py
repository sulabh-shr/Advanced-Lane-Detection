import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

from thresholds import *
from calibrate import get_camera_parameters, undistort
from perspective import warp
from search import search_lane, find_next_lane

if __name__ == '__main__':
    chess_rows = 6
    chess_columns = 9

    chess_regex_path = 'chessboards/chess*.jpg'
    chess_save_path = 'chessboards/undistorted/'

    test_regex_path = 'test_images/test*.jpg'
    test_save_path = 'test_images/undistorted/'

    thresh_s = (190, 255)
    thresh_sx = (40, 255)
    thresh_sy = (20, 110)
    thresh_dir = (0.68, 1.3)

    """
        Get camera parameter executed only once
        These remain constant for later use
        The parameters saved using pickle
    """
    # get_camera_parameters(chess_regex_path, chess_rows, chess_columns)
    with open("camera_params.txt", "rb") as f:
        ret, mtx, dist, rvecs, tvecs = pickle.load(f)

    # For testing the undistort output:
    # undistort(mtx, dist, chess_regex_path, chess_save_path)

    undistorted = undistort(mtx, dist, test_regex_path, test_save_path)
    binary_images = combined_threshold(undistorted, thresh_s, thresh_sx, thresh_sy, thresh_dir)

    i = 0
    for img in binary_images:
        a = warp(img, True, i)
        if i==0:
            l, r = search_lane(a, name=i)
        else:
            find_next_lane(a, l, r, name=i)
        i += 1

