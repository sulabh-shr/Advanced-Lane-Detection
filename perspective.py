import numpy as np
import cv2
from calibrate import read_images


def warp(image):
    src_high = 460
    src_low = 680
    src_low_x1 = 250
    src_low_x2 = 1030

    src = np.float32(
        [[530, src_high],
         [720, src_high],
         [src_low_x2, src_low],
         [src_low_x1, src_low]])

    dst_height = image.shape[0]
    dst_width = image.shape[1]
    offset = 200
    dst = np.float32(
        [[src_low_x1+offset, 0],
         [src_low_x2-offset, 0],
         [src_low_x2-offset, dst_height],
         [src_low_x1+offset, dst_height]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    warped_size = (dst_width, dst_height)
    warped = cv2.warpPerspective(image, M, warped_size, flags=cv2.INTER_LINEAR)
    cv2.imwrite('outputs/warped/' + 'warped.jpg', warped)
    cv2.imwrite('outputs/warped/' + 'img.jpg', image)
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
