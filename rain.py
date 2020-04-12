import cv2
import numpy as np
from matplotlib import pyplot as plt

histarray = []
for i in range(-1, 255):
    histarray.append(0)
img = cv2.imread('imgs/rain.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width = img.shape
for y in range(0, height):
    for x in range(0, width):
        histarray[img[y, x]] = histarray[img[y, x]]+1
cv2.imshow("dest", img)
f = plt.figure(1)
plt.plot(histarray)
f.show()
o = 0
for i in range(0, 255):
    if histarray[i] == 0:
        histarray.pop(i)
        histarray.append(0)
        o = o + 1
    else:
        break
for y in range(0, height):
    for x in range(0, width):
        if img[y, x] > 0:
            img[y, x] = img[y, x] - o
 #           print(i, y, x)
cv2.imshow("dest2", img)
ff = plt.figure(2)
plt.plot(histarray)
ff.show()
c = 0
for h in histarray:
    if h == 0:
        c = c + 1
    else:
        c = 0
inpmax = 255 - c

k = 255/inpmax

for y in range(0, height):
    for x in range(0, width):
        if img[y, x] * k > 255:
            img[y, x] = 255
        else:
            img[y, x] = int(img[y, x] * k)
cv2.imshow("dest3", img)

histarrayafter = []
for i in range(-1, 255):
    histarrayafter.append(0)

for y in range(0, height):
    for x in range(0, width):
#        print(img[y, x])
        histarrayafter[img[y, x]] = histarrayafter[img[y, x]]+1

fff = plt.figure(3)
plt.plot(histarrayafter)
fff.show()
print(np.max(img))
print(np.min(img))
cv2.waitKey(0)

