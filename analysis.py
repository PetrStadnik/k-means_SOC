import sys
import cv2
import numpy as np
import random as rnd
from matplotlib import pyplot as plt
from numba import jit
from sklearn import metrics
import time

k_arr = []
for i in range(1, 11):
    k = np.load("arrays/MQ/DBI/bgr"+str(i)+".npy")
    k_arr.append(np.where(min(k) == k)[0][0]+2)


#print(k_arr)
for i in range(1, 11):
    y1 = np.load("arrays/LL/Price/bgr" + str(i) + "k" + str(k_arr[i - 1]) + ".npy")[:]
    y2 = np.load("arrays/MQ/Price/bgr" + str(i) + "k" + str(k_arr[i - 1]) + ".npy")[:]
    #y1 = np.load("arrays/LL/Price/bgr" + str(i) + "k" + str(2) + ".npy")[:]
    #y2 = np.load("arrays/MQ/Price/bgr" + str(i) + "k" + str(2) + ".npy")[:]
    #print(str(y1) +"\n----------------------\n"+ str(y2))
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Průběh cenové funkce v posledních osmi iteracích u jednotlivých algoritmů')
    ax1.plot(y1, color="blue")
    ax2.plot(y2, color="orange")
    ax1.set_title("Lloydův algoritmus")
    ax2.set_title("MacQueenův algoritmus")
    ax1.grid(True)
    ax2.grid(True)
    #print(str(k_arr[i - 1]))

    plt.xticks(np.arange(0, len(y1), 1))

    #print(y1[-1] - y2[-1])
    #print((y1[-1] - y2[-1])*100/((y1[-1] + y2[-1])/2))
    print(str(i) + " & " + str(k_arr[i - 1]) + " & " + str(round(y1[-1], 3)) + " & " + str(round(y2[-1], 3)) + " & " + str(round(abs((y1[-1] - y2[-1])*100/((y1[-1] + y2[-1])/2)), 5))+"\\%" + " & " + str(len(y1)) + " & " + str(len(y2)) + "\\\\")
    #plt.show()
    #plt.savefig("graphs/prices/prices_bgr"+str(i)+".jpg", format="jpg")


cv2.waitKey()
