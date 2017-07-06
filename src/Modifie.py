import cv2
import os
from Input import load_data, IMAGE_SIZE

path = 'C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\zhangzhe\\'
classes = ()


def resize_with_pad(image, height, width):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image

if __name__ == '__main__':
    for files in os.listdir(path):
        filename = path + files
        img = cv2.imread(filename)
        img = resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
        cv2.imwrite(filename, img, (cv2.IMWRITE_PNG_COMPRESSION, 0))
        load_data(path, classes)