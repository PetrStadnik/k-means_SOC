import numpy as np
import cv2
import random as rnd
from numba import jit
from sklearn import metrics


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



@jit(nopython=True)
def createNewImage(img, c, inclus):
    colorArr = np.array([[228, 26, 28],
                         [55, 126, 184],
                         [77, 175, 74],
                         [152, 78, 163],
                         [255, 127, 0],
                         [255, 255, 51],
                         [166, 86, 40],
                         [247, 129, 191],
                         [153, 153, 153]])
    width = img.shape[1]
    x = 0
    y = 0

    if len(c.shape) > 1:
        for p in range(len(inclus)):  # array for new image
            if len(img.shape) == 2:
                img[y, x] = c[int(inclus[p])][:1]
                # img[y, x] = colorArr[int(inclus[p])][:1]  # chose if you want to use more contrast color
            else:
                img[y, x] = c[int(inclus[p])][:3]
                # img[y, x] = colorArr[int(inclus[p])][:3]  # chose if you want to use more contrast color
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


def kms(fn, nclus, maxIteration, allowBGR=True, positionWeight=None):
    if allowBGR == True:
        img = cv2.imread(fn)
    else:
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

    arr = prepareData(img, positionWeight)  # call function for preparing data
    c = initKMS(arr, nclus)  # call initialization function

    centroids_arr = []
    centroids_arr.append(c.copy())

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
    while delta > 0 and iterationNumber < maxIteration:  # set the criteria for stopping the algorithm
        iterationNumber += 1
        sumInClusterArr[:] = 0
        elementsInClusterArr[:] = 0
        sumOfAlldistance = 0
        delta = 0
        for p in range(len(arr)):
            i, dist = determineBestLabel(arr[p], c)  # call function for determining best label
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
        centroids_arr.append(c.copy())
        r_list.append(r)
        delta_list.append(delta / len(arr))
    createNewImage(img, c.astype(int), inclus)  # creating image after segmentation
    DB_index = metrics.davies_bouldin_score(arr, inclus)
    CH_index = metrics.calinski_harabasz_score(arr, inclus)
    return centroids_arr, DB_index, CH_index, img, sum_list, r_list, delta_list


if __name__ == "__main__":
    # Lloyd k-means algorithm outputs:      Array of centroids in each iteration
    #                                       Value of Davies-Bouldin index for segmentation
    #                                       Value of Calinski-Harabasz index for segmentation
    #                                       Image after segmentation
    #                                       Array of value of Price function in each iteration
    #                                       Array of maximal variance in centroids moving after each iteration
    #                                       Array of number of data-points changed cluster in each iteration
    centroids_arr, \
    DB_index, \
    CH_index, \
    img, \
    sum_list, \
    r_list, \
    delta_list \
        = kms(fn="some_image.jpg",    # Input image for segmentation
              maxIteration=300,     # Set maximal number of iteration to stop algorithm if it won't stop before
              nclus=3,              # Number of clusters
              allowBGR=True,        # Set segmentation in BGR spectre if True, or in BW spectre if False
              positionWeight=None)  # Set weight of position coordinates of pixels, None is equal 0
    cv2.waitKey(0)
