import sys
import cv2
import numpy as np
import random as rnd
from matplotlib import pyplot as plt
from numba import jit
from sklearn import metrics
import time


def initKMS(samples, nclus):
    if len(samples.shape) > 1:
        c = np.zeros(shape=(nclus, samples.shape[1]), dtype=np.float32)
    else:
        c = np.zeros(nclus, dtype=np.float32)
    tmp = len(samples)

    for i in range(nclus):
        con = True
        while con:
            j = rnd.randint(0, tmp - 1)
            conarr = np.isin(c, samples[j])
            con = np.any(conarr)
        c[i] = samples[j]
    print(c)
    return c


@jit(nopython=True)
def determineBestLabel(sample, centroids):
    i = 0
    dist = np.linalg.norm(sample - centroids[0])

    for p in range(1, len(centroids)):
        newDist = np.linalg.norm(sample - centroids[p])

        if dist > newDist:
            i = p
            dist = newDist

    return i, dist

def createGraph(xname, x, yname, y, title, savename, logver=False, showval=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, y)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    if showval:
        for i, v in enumerate(y):
            ax.text(i, v + 25, "%d" % v, ha="center")
    plt.savefig("graphs/test26/"+savename+".eps", format="eps")
    if logver:
        plt.clf()
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(title)
        plt.yscale("log")
        plt.plot(x, y)
        plt.savefig("graphs/test26/" + savename + "_log.eps", format="eps")
    plt.clf()

#@jit(nopython=True)
def createNewImage(img, c, inclus):
    colorArr = np.array([[228,26,28],
                        [55,126,184],
                        [77,175,74],
                        [152,78,163],
                        [255,127,0],
                        [255,255,51],
                        [166,86,40],
                        [247,129,191],
                        [153,153,153]])
    width = img.shape[1]
    x = 0
    y = 0
    print(c)

    if len(c.shape) > 1:
        for p in range(len(inclus)):  # array for new image
            if len(img.shape) == 2:
                #img[y, x] = c[int(inclus[p])][:1]
                img[y, x] = colorArr[int(inclus[p])][:1]
            else:
                #img[y, x] = c[int(inclus[p])][:3]
                img[y, x] = colorArr[int(inclus[p])][:3]
            x = x + 1
            if x == width:
                y = y + 1
                x = 0
    else:
        for p in range(len(inclus)):  # array for new image
            img[y, x] = c[int(inclus[p])]
            x = x + 1
            if x == width:
                y = y + 1
                x = 0
    cv2.imwrite("savedImages/test26/bgr" + str(n) + "_c" + str(c[1]) + str(c[2]) + ".jpg", img)
    return img


def prepareData(img, positionWeight):
    if len(img.shape) == 2:
        height, width = img.shape
        if positionWeight != None:
            arr = np.zeros(shape=(height * width, 3), dtype=np.float32)
            imgArr = img.flatten()
            x = 0
            y = 0
            for i in range(height * width):
                arr[i, 0] = imgArr[i]
                if x == width:
                    x = 0
                    y += 1
                arr[i, 1] = x * positionWeight
                arr[i, 2] = y * positionWeight
                x += 1
        else:
            arr = np.array(img.flatten(), dtype=np.float32)

    else:
        height, width, channels = img.shape
        b, g, r, = cv2.split(img)
        b = b.flatten()
        g = g.flatten()
        r = r.flatten()

        # create data matrix
        if positionWeight != None:
            arr = np.zeros(shape=(height * width, channels + 2), dtype=np.float32)
            x = 0
            y = 0
            for i in range(height * width):
                arr[i, 0] = b[i]
                arr[i, 1] = g[i]
                arr[i, 2] = r[i]
                if x == width:
                    x = 0
                    y += 1
                arr[i, 3] = x * positionWeight
                arr[i, 4] = y * positionWeight
                x += 1
        else:
            arr = np.array(img, dtype=np.float32).reshape(height * width, channels)
    return arr


def kms(fn, nclus, stop=1e-5, maxIteration=300, allowBGR=True, positionWeight=None):
    if allowBGR == True:
        img = cv2.imread(fn)
    elif allowBGR == False:
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

    arr = prepareData(img, positionWeight)
    c = initKMS(arr, nclus)
    shots = c.copy()
    inclus = -1 * np.ones(img.shape[0] * img.shape[1], dtype=int)  # array in size arr for determinate nearest centroid
    sum_list = []
    r_list = []
    delta_list = []
    r = 1
    if len(arr.shape) > 1:
        sumInClusterArr = np.zeros(shape=(nclus, arr.shape[1]))  # array for count element values in each cluster
        elementsInClusterArr = np.zeros(shape=(nclus, arr.shape[1]))  # array for number of elements in cluster
    else:
        sumInClusterArr = np.zeros(nclus)  # array for count element values in each cluster
        elementsInClusterArr = np.zeros(nclus)  # array for number of elements in cluster
    iterationNumber = 0
    delta = 1
    # while r > stop and iterationNumber < maxIteration:
    while delta > 0 and iterationNumber < maxIteration:
        print("delta: " + str(delta / len(arr)))
        iterationNumber += 1
        sumInClusterArr[:] = 0
        elementsInClusterArr[:] = 0
        sumOfAlldistance = 0
        delta = 0
        for p in range(len(arr)):
            i, dist = determineBestLabel(arr[p], c)
            if inclus[p] != i:
                delta += 1
            inclus[p] = i
            sumInClusterArr[i] += arr[p]  # sum of element values in one cluster
            elementsInClusterArr[i] += 1  # number of elements in cluster

            sumOfAlldistance += dist

        sum_list.append(sumOfAlldistance)
        old_r = r
        r = 0
        for p in range(nclus):
            newCentroid = sumInClusterArr[p] / elementsInClusterArr[p]  # compute new centroids
            oldNewCentroidDistance = np.linalg.norm(c[p] - newCentroid)

            if oldNewCentroidDistance > r:
                r = oldNewCentroidDistance
            c[p] = newCentroid
            if old_r == r:
                iterationNumber = maxIteration
        r_list.append(r)
        delta_list.append(delta / len(arr))
        print("r: " + str(r))
        createNewImage(img, c.astype(int), inclus)
    print("Iteration number:")
    print(iterationNumber)
    #createNewImage(img, c.astype(int), inclus)
    DB_index = metrics.davies_bouldin_score(arr, inclus)
    CH_index = metrics.calinski_harabasz_score(arr, inclus)
    return shots, DB_index, img, sum_list, r_list, delta_list


if __name__ == "__main__":


    for n in range(1, 2):
        dbi_arr = []
        f = open("testLog/test26.txt", "a")
        f.write("\n -------------------------*****************-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-\n bgr: " + str(n))
        for k in range(5, 6):
            sttime = time.time()
            shots, DB_index, img, sum_list, r_list, delta_list = kms(fn="imgs/BGR/bgr"+ str(n) +".jpg", maxIteration=300, nclus=k, allowBGR=True,
                                                              positionWeight=None)
            createGraph(xname="iterace", x=range(len(sum_list)), yname="cenová funkce", y=sum_list, title="Cenová funkce",
                        savename="cenova_bgr"+str(n)+"k"+ str(k), logver=True)
            createGraph(xname="iterace", x=range(len(sum_list)), yname="delta funkce", y=delta_list,
                        title="Delta funkce", savename="delta_bgr"+str(n)+"k"+ str(k))
            createGraph(xname="iterace", x=range(len(sum_list)), yname="R funkce", y=r_list,
                        title="R funkce", savename="r_bgr"+str(n)+"k"+ str(k))
            dbi_arr.append(DB_index)
            print("DB_index  " + str(DB_index))
            cv2.imwrite("savedImages/test26/bgr"+str(n)+"_k" + str(k) + ".jpg", img)
            f.write("\n ------------------------------------------\n k: " + str(k))
            f.write("\n nastrel: " + str(shots))
            f.write("\n DB: " + str(DB_index))
            f.write("\n time: " + str(time.time()-sttime))
            f.write("\n pocet iteraci: " + str(len(r_list)))
            f.write("\n cenova funkce: " + str(sum_list))
            f.write("\n delta funkce: " + str(delta_list))
        createGraph(xname="počet shluků", x=range(2, len(dbi_arr)+2), yname="DBI", y=dbi_arr, title="Davis-Bouldin Index", savename="DBIbgr"+str(n),
                    showval=True)
        f.close()

    cv2.waitKey(0)
