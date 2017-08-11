import cv2
import numpy as np


def threshold_hls(images, t):
    output = []

    for image in images:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s = hls[:, :, 2]
        selected = np.zeros_like(s)  # all pixels 0
        selected[(s > t[0]) & (s <= t[1])] = 1  # selected pixels 1
        output.append(selected)

    i = 0
    for img in output:
        temp = np.zeros_like(img)
        temp[img == 1] = 255
        cv2.imwrite('outputs/' + str(i) + 's.jpg', temp)
        cv2.imwrite('outputs/s' + str(i) + 's.jpg', temp)
        i += 1

    return output


def threshold_sobelx(images, t):
    output = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sobelx --> Absolute --> Scaling to range [0-255]
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        selected = np.zeros_like(sobelx)  # all pixels 0
        selected[(sobelx > t[0]) & (sobelx <= t[1])] = 1  # selected pixels 1
        output.append(selected)

    i = 0
    for img in output:
        temp = np.zeros_like(img)
        temp[img == 1] = 255
        cv2.imwrite('outputs/' + str(i) + 'sx.jpg', temp)
        cv2.imwrite('outputs/sx' + str(i) + 'sx.jpg', temp)
        i += 1

    return output


def threshold_sobely(images, t):
    output = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sobelx --> Absolute --> Scaling to range [0-255]
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobely = np.absolute(sobely)
        sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
        selected = np.zeros_like(sobely)  # all pixels 0
        selected[(sobely > t[0]) & (sobely <= t[1])] = 1  # selected pixels 1
        output.append(selected)

    i = 0
    for img in output:
        temp = np.zeros_like(img)
        temp[img==1] = 255
        cv2.imwrite('outputs/' + str(i) + 'sy.jpg', temp)
        cv2.imwrite('outputs/sy' + str(i) + 'sy.jpg', temp)
        i += 1

    return output


def threshold_sobel_dir(images, t):
    output = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

        abs_sobely = np.absolute(sobely)
        abs_sobelx = np.absolute(sobelx)

        angles = np.arctan2(abs_sobely, abs_sobelx)

        selected = np.zeros_like(angles)  # all pixels 0
        selected[(angles > t[0]) & (angles <= t[1])] = 1  # selected pixels 1
        output.append(selected)

    i = 0
    for img in output:
        temp = np.zeros_like(img)
        temp[img==1] = 255
        cv2.imwrite('outputs/' + str(i) + 'dir.jpg', temp)
        cv2.imwrite('outputs/dir' + str(i) + 'dir.jpg', temp)
        i += 1

    return output


def combined_threshold(images, ts, tsx, tsy, td):
    output = []
    s_channel = threshold_hls(images, ts)
    sobel_x = threshold_sobelx(images, tsx)
    sobel_y = threshold_sobely(images, tsy)
    sobel_dir = threshold_sobel_dir(images, td)
    i = 0
    for s, sx, sy, sd in zip(s_channel, sobel_x, sobel_y, sobel_dir):
        selected = np.zeros_like(s)

        c1 = (s==1)
        c2 = (sx==1) & (sd==1)

        selected[c1 | c2] = 1
        output.append(selected)
        to_write = np.zeros_like(selected)
        to_write[selected==1] = 255
        cv2.imwrite('outputs/' + str(i) + 'cmb.jpg', to_write)
        cv2.imwrite('outputs/cmb' + str(i) + 'cmb.jpg', to_write)
        i+=1
    return output

