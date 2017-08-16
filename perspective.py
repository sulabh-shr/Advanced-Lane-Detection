import numpy as np
import cv2
from calibrate import read_images


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
        save_img = np.zeros_like(warped)
        save_img[warped==1] = 255
        cv2.imwrite('outputs/warped/' + str(name)+'warped.jpg', warped)
        save_img = np.zeros_like(image)
        save_img[image==1] = 255
        cv2.imwrite('outputs/warped/' + str(name)+'img.jpg', image)
    return warped

    # for i in images:
    #     plt.imshow(i)
    #     plt.plot(530, 415, '.')
    #     plt.plot(720, 415, '.')
    #     plt.plot(200, 680, '.')
    #     plt.plot(1200, 680, '.')
    #     plt.show()

if __name__ == '__main__':
    images = read_images('test_images/test*.jpg')
    i = 0
    for img in images:
        warp(img)
        i += 1
