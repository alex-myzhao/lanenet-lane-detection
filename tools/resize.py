import cv2
import os
from os import path
import glob


def main():
    # input folder
    files = [f for f in glob.glob('./gt_image_binary_ori/*.png')]
    for f in files:
        img = cv2.imread(f)
        img = img[200:, :]
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path.join('./gt_image_binary', path.basename(f)), img)


if __name__ == '__main__':
    main()
