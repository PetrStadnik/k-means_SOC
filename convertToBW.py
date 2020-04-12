import cv2
for p in range(1, 5):
    print(p)
    img = cv2.imread("imgs/p/bgrp" + str(p) + ".jpg")
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = 600
    height = int(img.shape[0] * ((width * 100)/img.shape[1]) / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #cv2.imwrite("imgs/BW/bw" + str(p) + '.jpg', resized)
    cv2.imwrite("imgs/BGR/bgr" + str(p) + '.jpg', resized)
cv2.waitKey(0)
