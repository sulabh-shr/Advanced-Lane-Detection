import glob
import numpy as np
import cv2
import pickle


def read_images(path):
    images = []
    paths = glob.glob(path)
    for path in paths:
        images.append(cv2.imread(path))
    return images


def get_camera_parameters(reg_path, rows, columns):
    images = read_images(reg_path)
    paths = glob.glob(reg_path)
    image_pts = []  # 2D coordinates of images
    object_pts = []  # 3D coordinates of images in the world
    object_pt = np.zeros((rows * columns, 3), np.float32)  # predefined 3D coord of 1 image
    object_pt[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    for img, image_path in zip(images, paths):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (columns, rows), None)
        print(found, image_path)

        if found:
            image_pts.append(corners)
            object_pts.append(object_pt)
            combined_img = cv2.drawChessboardCorners(img, (columns, rows), corners, found)
            cv2.imwrite(image_path.split('/')[0]+'/detected/cb_' + image_path.split('/')[-1], combined_img)

    h, w = gray.shape[:2]
    with open('camera_params.txt', 'wb') as f:
        pickle.dump(cv2.calibrateCamera(object_pts, image_pts, (w, h), None, None), f)


def undistort(mtx, dist, image_path, save_path):
    images = read_images(image_path)
    paths = glob.glob(image_path)
    output = []
    for img, path in zip(images, paths):
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        output.append(undistorted)
        cv2.imwrite(save_path + 'un_' + path.split('/')[-1], undistorted)
    return output