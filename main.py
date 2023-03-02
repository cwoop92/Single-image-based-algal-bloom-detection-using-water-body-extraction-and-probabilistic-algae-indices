import cv2
import matplotlib.pyplot as plt
import time
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import spdscan as spd
import aslic
import wavent as wv

img= cv2.imread("6.jpg")
ratio = 800 / max([img.shape[1], img.shape[0]])
if ratio < 1:
    img = cv2.resize(img, (round(img.shape[1] * ratio), round(img.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
numSegments = 3000
start_time = time.time()
segments = slic(img, numSegments, sigma = 5)

fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img, segments))
plt.axis("off")

Am = aslic.regionadjacency(segments)

C  = aslic.makeStruct(segments, labimg)

lc = spd.spdbscan(segments, C, Am, 12)

mxIm, mxLc = spd.maxLabel(lc, img)

dataCal = wv.waveEnt(img, mxLc)

dataV = wv.hsvWeight(img, mxLc)

imRiver = wv.mapRiver(img, mxIm, mxLc, dataCal, dataV)

end_time = time.time()
print("--- %s seconds ---" %(end_time - start_time))

plt.imshow(imRiver)
plt.show()

cv2.imwrite("out.jpg", imRiver)
