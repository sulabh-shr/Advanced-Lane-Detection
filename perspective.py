import numpy as np
import cv2
from calibrate import read_images
import matplotlib.pyplot as plt

def warp(image, save=False, name=''):
    src_high = 460
    src_low = 700
    src_low_x1 = 220
    src_low_x2 = 1110

    src = np.float32(
        [[570, src_high],
         [710, src_high],
         [src_low_x2, src_low],
         [src_low_x1, src_low]])

    dst_height = image.shape[0]
    dst_width = image.shape[1]
    offset = 250
    dst = np.float32(
        [[offset, 0],
         [dst_width-offset, 0],
         [dst_width-offset, dst_height],
         [offset, dst_height]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    warped_size = (dst_width, dst_height)
    warped = cv2.warpPerspective(image, M, warped_size, flags=cv2.INTER_LINEAR)
    if save:
        plt.imshow(warped, cmap='gray')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('outputs/warped/' + str(name) + 'warped.jpg')
        plt.close()
        plt.imshow(image, cmap='gray')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('outputs/warped/' + str(name) + 'img.jpg')
        plt.close()
    return warped


if __name__ == '__main__':
    images = read_images('test_images/test*.jpg')
    i = 0
    for img in images:
        warp(img)
        i += 1
