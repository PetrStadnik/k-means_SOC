import cv2
import numpy as np
from matplotlib import pyplot as plt
p = 1
while p != 0:
    p = int(input("Zadej prah 1-255"))
    img = cv2.imread('imgs/booktext.jpg')
    cv2.imshow("Puvodni letadlo", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    for y in range(0, height):
        for x in range(0, width):
            if img[y, x] > p:
                img[y, x] = 255
            else:
                img[y, x] = 0
    cv2.imshow("Letadlo!", img)
    cv2.imwrite("imgs/prah" + str(p) + '_nbooktext.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv2.waitKey(0)
