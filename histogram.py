import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('imgs/kmstennis.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
histarray = []
histarrayafter = []

for i in range(-1, 255):
    histarray.append(0)
    histarrayafter.append(0)

height, width = img.shape
inmin = np.min(img)
inmax = np.max(img)
outmax = int(input("Zadej číslo:"))
k = outmax/(inmax-inmin)

for y in range(0, height):
    for x in range(0, width):
        histarray[img[y, x]] = histarray[img[y, x]]+1
f = plt.figure(1)
plt.plot(histarray)
f.show()
cv2.imshow("před roztažením", img)
for y in range(0, height):
    for x in range(0, width):
        if (img[y, x] - inmin) * k > 255:
            img[y, x] = 255
        else:
            img[y, x] = int((img[y, x] - inmin) * k)

for y in range(0, height):
    for x in range(0, width):
        histarrayafter[img[y, x]] = histarrayafter[img[y, x]]+1

ff = plt.figure(2)
plt.plot(histarrayafter)
ff.show()
cv2.imshow("Po roztažení", img)
print(inmin, inmax, outmax)
cv2.waitKey(0)