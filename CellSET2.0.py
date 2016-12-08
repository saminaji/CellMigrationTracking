# menu.py


import tkMessageBox
import pygubu
import zipfile
try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from numpy.linalg import inv
from Tkinter import *
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
from tkFileDialog import asksaveasfilename
from tkFileDialog import asksaveasfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkMessageBox
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import exposure

from pylab import *
from numpy import *
import scipy
import operator
from EGBS import *
import os, csv, sys
import cv2
from PIL import Image as img2
from PIL import ImageSequence
from Tkinter import Image
import numpy as np
from libtiff import TIFF

import time
import ttk
from IPython import get_ipython
import ImageTk
import mahotas



tmppath = os.getcwd()
tmppath = os.path.join(tmppath, time.strftime("%d_%m_%Y_%I"))



if os.path.exists(tmppath) is False:
    os.mkdir(tmppath)

# create directories

trackingdir = os.path.join(tmppath + os.sep, 'tracking')
trajectorydir = os.path.join(tmppath + os.sep, 'finalplot')
overlaytrajectorydir = os.path.join(tmppath + os.sep, 'overlaytrajecory')
overlaytrajectoryanidir = os.path.join(tmppath + os.sep, 'overlaytrajecoryani')
masktrajectorydir = os.path.join(tmppath + os.sep, 'masktrajector')
csvdir = os.path.join(tmppath + os.sep, 'datafiles')


if os.path.exists(trackingdir) is False:
    os.mkdir(trackingdir)

if os.path.exists(trajectorydir) is False:
    os.mkdir(trajectorydir)

if os.path.exists(csvdir) is False:
    os.mkdir(csvdir)

if os.path.exists(overlaytrajectorydir) is False:
    os.mkdir(overlaytrajectorydir)

if os.path.exists(masktrajectorydir) is False:
    os.mkdir(masktrajectorydir)

if os.path.exists(overlaytrajectoryanidir) is False:
    os.mkdir(overlaytrajectoryanidir)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(20, 20), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))

# Create some random colors

color = np.random.randint(0, 255, (1000, 3))




def read_others(self, path):
    frames = []
    for frame in path:
        frames = cv2.imread(frame)
    return frames

def animate(self, trackingdir):
    # CHECK IF LIST IS EMPTY
    if len(self.gifBackgroundImages) == 0:

        # CREATE FILES IN LIST
        for foldername in os.listdir(trackingdir):
            self.gifBackgroundImages.append(foldername)

        self.gifBackgroundImages.sort(key=lambda x: int(x.split('.')[0]))

    if self.atualGifBackgroundImage == len(self.gifBackgroundImages):
        self.atualGifBackgroundImage = 0
    try:
        self.background["file"] = trackingdir + self.gifBackgroundImages[self.atualGifBackgroundImage]
        self.label1["image"] = self.background
        self.atualGifBackgroundImage += 1
    except EOFError:
        print (trackingdir + self.gifBackgroundImages[self.atualGifBackgroundImage])
        pass

    # MILISECONDS\/ PER FRAME
    self.after(300, lambda: animate(self, trackingdir))


def morph_dilate(self, image):
    image = cv2.dilate(image, cv2.MORPH_DILATE, kernel)
    return image


def morph_close(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def morph_open(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def morph_gradient(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return image


def morph_erode(self, image):
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
    return image


def white_background(image, kernel):

    im = cv2.threshold(image, 173, 255, cv2.THRESH_BINARY)
    im = im[1]
    dilation = cv2.dilate(im, kernel, iterations=1)
    gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)


    shifted = cv2.pyrMeanShiftFiltering(closing, 10, 20)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    # create a mask
    mask2 = np.zeros(image.shape, dtype="uint8")

    #  loop over the unique labels returned by the Watershed  algorithm for
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask2[labels == label] = 255
        # close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    return mask2


def basic_seg(img, images):
    noOfFrames = len(images)
    bgFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1, 4):
        bgFrame = bgFrame / 2 + \
                  cv2.cvtColor((images[i]),
                               cv2.COLOR_BGR2GRAY) / 2

    # Array to save the object locations
    objLocs = np.array([None, None])

    # Kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Display the frames like a video
    # Read each frame
    frame = images[1]

    # Perform background subtraction after median filter
    diffFrame = cv2.absdiff(cv2.cvtColor(cv2.medianBlur(frame, 7), \
                                         cv2.COLOR_BGR2GRAY), cv2.medianBlur(bgFrame, 7))

    # Otsu thresholding to create the binary image
    [th, bwFrame] = cv2.threshold(diffFrame, 0, 255, cv2.THRESH_OTSU)

    # Morphological opening operation to remove small blobs
    bwFrame = cv2.morphologyEx(bwFrame, cv2.MORPH_OPEN, kernel)

    return bwFrame

def black_background(image, kernel):

    shifted = cv2.pyrMeanShiftFiltering(image, 10, 39)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # create a mask
    mask2 = np.zeros(gray.shape, dtype="uint8")
    #  loop over the unique labels returned by the Watershed  algorithm for
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask2[labels == label] = 255
    return mask2


# histgram equalization
def histogram_equaliz(image):
    old_gray_image = cv2.equalizeHist(image)
    return old_gray_image


def shi_tomasi(image, maxCorner, qualityLevel, MinDistance):
    # detect corners in the image

    corners = cv2.goodFeaturesToTrack(image,
                                      maxCorner,
                                      qualityLevel,
                                      MinDistance,
                                      mask=None,
                                      blockSize=7)

    return corners


def harris_corner(image, maxCorner, qualityLevel, minDistance):
    corners = cv2.goodFeaturesToTrack(image,  # img
                                      maxCorner,  # maxCorners
                                      qualityLevel,  # qualityLevel
                                      minDistance,  # minDistance
                                      None,  # corners,
                                      None,  # mask,
                                      7,  # blockSize,
                                      useHarrisDetector=True,  # useHarrisDetector,
                                      k=0.05  # k
                                      )
    return corners

# find blobs in images
def blob_seg(image):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 20
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.50

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    # Filter by Color
    params.blobColor = 255
    params.filterByColor = True
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    keypoints = detector.detect(image)

    pts = [[[np.float32(np.hstack(keypoints[idx].pt))]] for idx in range(len(keypoints))]
    pts = np.vstack(pts)

    return pts, keypoints

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = numpy.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 15 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

# find contour
def contour(file):
    img = file
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    LENGTH = len(contours)
    status = numpy.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in xrange(maximum):
        pos = numpy.where(status == i)[0]
        if pos.size != 0:
            cont = numpy.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    cv2.drawContours(image, unified, -1, (0, 255, 0), 2)

    cv2.imshow("window title", image)
    cv2.waitKey(0)


## Segment color image, using the 'Efficient Graph Based Image Segmentation' (EGBS) and show the results
def segmentColorImageEGBS(self, file):

    gray2 = cv2.cvtColor(file, cv2.COLOR_RGB2GRAY)
    image = exposure.equalize_hist(file)


    gray2 = cv2.cvtColor(file, cv2.COLOR_RGB2GRAY)
    hist = cv2.equalizeHist(gray2)

    h, w, d = image.shape
    print 'image.shape: ', image.shape
    imshow(image); title('Input image'); draw()

    TH = 1200    # threshold

    egbs = EGBS(w, h, threshold=TH, minSize=30)
    egbs.segmentImage(image)
    print 'Merge small components'
    egbs.mergeSmall()
    print 'number of segments: ', egbs.numSets()

    #labels = egbs.getLabels()
    labels, edges = egbs.getSegmentEdges()
    gray()
    #matshow(edges);    title('edges');    draw()
    mask = numpy.zeros(gray2.shape, dtype="uint8")
    for label in range(len(unique(labels))):
        if label ==0:
            continue
        mask[labels == label] = 255

    cv2.imshow("Keypoints", mask)
    cv2.waitKey(33)
    contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]



    return pts, contours


def basic_tracking(self, frames, old_gray_image1, intialPoints, segMeth, maxCorner, qualityLevel, minDistance,
                 updateconvax, progessbar, timelapse, line = False, ID = False):

    old_gray_image2 = cv2.cvtColor(old_gray_image1, cv2.COLOR_BGR2GRAY)
    old_gray_image2 = histogram_equaliz(old_gray_image2)

    feature_params = dict(maxCorners=maxCorner,
                          qualityLevel=qualityLevel,
                          minDistance=minDistance,
                          blockSize=7)

    Initialtime = timelapse
    mask = np.zeros_like(old_gray_image1,)

    finalFrame = len(frames)

    # training a knn model
    firstDetections, updatedTrackIdx, updateDetections,old_trackIdx = [], [], [], []
    idx = 0
    for _, row in enumerate(intialPoints):
        g, d = row.ravel()
        firstDetections.append([g, d])
        updateDetections.append([g, d])
        updatedTrackIdx.append(idx)
        old_trackIdx.append(idx)
        idx += 1

    firstDetections = np.vstack(firstDetections)
    updateDetections = np.vstack(updateDetections)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(firstDetections)

    mask = np.zeros_like(old_gray_image1, )
    trajectoriesX, trajectoriesY, cellIDs, frameID, t = [], [], [], [], []

    for i, frame in enumerate(frames):
        try:

            if segMeth == 'blob':
                p1,_ = blob_seg(frame)
            else:
                new_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # perform histogram equalization to balance image color intensity
                new_gray_image = histogram_equaliz(new_gray_image)

                p1 = cv2.goodFeaturesToTrack(new_gray_image, mask=None, **feature_params)

            good_new = []

            for _, row in enumerate(p1):
                C2, D2 = row.ravel()
                good_new.append([C2, D2])
            # make the given detection matrix

            good_new = np.vstack(good_new)

            secondDetections = updateDetections

            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(updateDetections)

            updatedTrackIdx = []
            for tt, row in enumerate(good_new):
                z, y = row.ravel()
                test = np.hstack([z, y])
                # find the closet point to the testing  in the training data
                nearestpoint = neigh.kneighbors(test)
                trackID = int(nearestpoint[1][0])
                distance = nearestpoint[0][0]
                distance = np.float32(distance[0])

                if distance > 15:
                    new_idx = old_trackIdx[-1] + 1
                    updatedTrackIdx.append(new_idx)
                    old_trackIdx.append(new_idx)
                    updateDetections = np.vstack([updateDetections, test])
                    secondDetections = np.vstack([secondDetections, test])
                else:
                    updatedTrackIdx.append(trackID)
                    updateDetections[trackID] = np.hstack(test)

            cont = len(good_new)

            for ii, new in enumerate(good_new):
                cellId = updatedTrackIdx[ii]

                a, b = new.ravel()

                c, d = secondDetections[cellId]

                if line == True:
                    mask = cv2.line(mask, (a, b), (c, d), color[cellId].tolist(), 2)
                else:
                    mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)

                if ID == True:
                    frame = cv2.putText(frame, "%d" % cellId, (c, d), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
                else:
                    frame = cv2.putText(frame, " ", (c, d), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        color[cellId].tolist())

                img = cv2.add(frame, mask)
                edged2 = np.hstack([mask, img])


                intialPoints = good_new.reshape(-1, 1, 2)
                # Keep the data of for later processing
                trajectoriesX.append(c)
                trajectoriesY.append(d)
                cellIDs.append(int(cellId))
                frameID.append(i)
                t.append(timelapse)

                # manage the displaying label
                if ii == 0:
                    self.label_10.configure(text=int(timelapse))
                    self.label_43.configure(text=a)
                    self.label_63.configure(text=b)
                if ii == 1:
                    self.label_11.configure(text=int(timelapse))
                    self.label_44.configure(text=a)
                    self.label_64.configure(text=b)

                if ii == 2:
                    self.label_12.configure(text=int(timelapse))
                    self.label_45.configure(text=a)
                    self.label_65.configure(text=b)

                if ii == 3:
                    self.label_13.configure(text=int(timelapse))
                    self.label_46.configure(text=a)
                    self.label_66.configure(text=b)

                if ii == 4:
                    self.label_14.configure(text=int(timelapse))
                    self.label_47.configure(text=a)
                    self.label_67.configure(text=b)

                if ii == 5:
                    self.label_15.configure(text=int(timelapse))
                    self.label_48.configure(text=a)
                    self.label_68.configure(text=b)
                if ii == 6:
                    self.label_16.configure(text=int(timelapse))
                    self.label_49.configure(text=a)
                    self.label_69.configure(text=b)
                if ii == 7:
                    self.label_17.configure(text=int(timelapse))
                    self.label_50.configure(text=a)
                    self.label_70.configure(text=b)

                if ii == 8:
                    self.label_18.configure(text=int(timelapse))
                    self.label_51.configure(text=a)
                    self.label_71.configure(text=b)

                if ii == 9:
                    self.label_19.configure(text=int(timelapse))
                    self.label_52.configure(text=a)
                    self.label_72.configure(text=b)

            r = 500.0 / img.shape[1]
            dim = (500, int(img.shape[0] * r))

            # perform the actual resizing of the image and show it
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            mahotas.imsave(os.path.join(trackingdir, '%d.gif' % i), img)
            mahotas.imsave(os.path.join(masktrajectorydir, '%d.gif' % i), mask)
            mahotas.imsave(os.path.join(overlaytrajectorydir, '%d.gif' % i), resized)

            tmp_path = os.path.join(trackingdir, '%d.gif' % i)
            image = img2.open(str(tmp_path))
            image = ImageTk.PhotoImage(image)
            root.image = image
            imagesprite = updateconvax.create_image(295, 190, image=image, anchor='c')
            time.sleep(1)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            if i == finalFrame - 1:
                cv2.imwrite(os.path.join(trajectorydir, 'finalTrajectory.png'), img)
                cv2.imwrite(os.path.join(trajectorydir, 'Plottrajector.png'), mask)
                image = img2.open(str(tmp_path))
                image = ImageTk.PhotoImage(image)
                root.image = image
                imagesprite4 = updateconvax.create_image(263, 187, image=image)

        except EOFError:
            continue
        timelapse += Initialtime

        unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY, t)
        with open(os.path.join(csvdir, 'data.csv'), 'wt') as f1:
            writer = csv.writer(f1, lineterminator='\n')
            writer.writerow(('frameID', 'track_no', 'x', "y", "time",))
            for value in unpacked:
                writer.writerow(value)

def optical_flow(self, frames, old_gray_image1, intialPoints, segMeth, maxCorner, qualityLevel, minDistance,
                 updateconvax, progessbar, timelapse, line = False, ID = False):

    old_gray_image2 = cv2.cvtColor(old_gray_image1, cv2.COLOR_BGR2GRAY)
    old_gray_image2 = histogram_equaliz(old_gray_image2)


    Initialtime = timelapse
    mask = np.zeros_like(old_gray_image1,)

    finalFrame = len(frames)

    # training a knn model
    firstDetections, updatedTrackIdx, updateDetections,old_trackIdx = [], [], [], []
    idx = 0
    print intialPoints

    for row in intialPoints:
        r1 = row[0]
        g, d = r1.ravel()
        firstDetections.append(np.float32([g, d]))
        updateDetections.append(np.float32([g, d]))
        updatedTrackIdx.append(idx)
        old_trackIdx.append(idx)
        idx += 1

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(firstDetections)

    trajectoriesX, trajectoriesY, cellIDs, frameID, t = [], [], [], [],[]

    for i, frame in enumerate(frames):
        try:
            new_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # perform histogram equalization to balance image color intensity
            new_gray_image = histogram_equaliz(new_gray_image)

            # show the progress bar
            progessbar.step(i*2)
            initialValue = 0

            newPoints, state, errorRate = cv2.calcOpticalFlowPyrLK(old_gray_image2, new_gray_image, intialPoints, None, **lk_params)

            # Select good points

            good_new = newPoints[state == 1]
            good_old = intialPoints[state == 1]

            secondDetections = updateDetections
            good_new2 = good_new
            if i > 0:
                neigh = NearestNeighbors(n_neighbors=1)
                neigh.fit(updateDetections)

                updatedTrackIdx = []
                for tt, row in enumerate(good_new):
                    z, y = row.ravel()
                    test = np.hstack(np.float32([z, y]))
                    # find the closet point to the testing  in the training data
                    nearestpoint = neigh.kneighbors(test)
                    trackID = int(nearestpoint[1][0])
                    distance = float(nearestpoint[0][0])



                    if not distance:
                        good_new2 = np.delete(good_new, tt, 0)
                    else:
                        good_new2 = []
                        good_new2 = good_new

                    if distance > 15:
                        new_idx = old_trackIdx[-1] + 1
                        updatedTrackIdx.append(new_idx)
                        old_trackIdx.append(new_idx)
                        updateDetections = np.vstack([updateDetections, test])
                        secondDetections = np.vstack([secondDetections, test])
                    else:
                        updatedTrackIdx.append(trackID)
                        updateDetections[trackID] = test


            for ii, new in enumerate(good_new2):

                cellId = updatedTrackIdx[ii]

                a, b = new.ravel()

                c, d = secondDetections[cellId]


                if line == True:
                    mask = cv2.line(mask, (a, b), (c, d), color[cellId].tolist(), 2)
                else:
                    mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)

                if ID == True:
                    frame = cv2.putText(frame, "%d" % cellId, (c, d), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0))
                else:
                    frame = cv2.putText(frame, " " , (c, d), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                            color[cellId].tolist())

                img = cv2.add(frame, mask)
                edged2 = np.hstack([mask, img])

                # Now update the previous frame and previous points
                old_gray_image2 = new_gray_image.copy()

                intialPoints = good_new.reshape(-1, 1, 2)
                # Keep the data of for later processing
                trajectoriesX.append(c)
                trajectoriesY.append(d)
                cellIDs.append(int(cellId))
                frameID.append(i)
                t.append(timelapse)

                # manage the displaying label
                if ii ==0:
                    self.label_10.configure(text=int(timelapse))
                    self.label_43.configure(text=a)
                    self.label_63.configure(text=b)
                if ii == 1:
                    self.label_11.configure(text=int(timelapse))
                    self.label_44.configure(text=a)
                    self.label_64.configure(text=b)

                if ii == 2:
                    self.label_12.configure(text=int(timelapse))
                    self.label_45.configure(text=a)
                    self.label_65.configure(text=b)

                if ii == 3:
                    self.label_13.configure(text=int(timelapse))
                    self.label_46.configure(text=a)
                    self.label_66.configure(text=b)

                if ii == 4:
                    self.label_14.configure(text=int(timelapse))
                    self.label_47.configure(text=a)
                    self.label_67.configure(text=b)

                if ii == 5:
                    self.label_15.configure(text=int(timelapse))
                    self.label_48.configure(text=a)
                    self.label_68.configure(text=b)
                if ii == 6:
                    self.label_16.configure(text=int(timelapse))
                    self.label_49.configure(text=a)
                    self.label_69.configure(text=b)
                if ii == 7:
                    self.label_17.configure(text=int(timelapse))
                    self.label_50.configure(text=a)
                    self.label_70.configure(text=b)

                if ii == 8:
                    self.label_18.configure(text=int(timelapse))
                    self.label_51.configure(text=a)
                    self.label_71.configure(text=b)

                if ii == 9:
                    self.label_19.configure(text=int(timelapse))
                    self.label_52.configure(text=a)
                    self.label_72.configure(text=b)



            r = 500.0 / img.shape[1]
            dim = (500, int(img.shape[0] * r))

            # perform the actual resizing of the image and show it
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            mahotas.imsave(os.path.join(trackingdir, '%d.gif' % i), img)
            mahotas.imsave(os.path.join(masktrajectorydir, '%d.gif' % i), mask)
            mahotas.imsave(os.path.join(overlaytrajectorydir, '%d.gif' % i), resized)

            tmp_path = os.path.join(trackingdir, '%d.gif' % i)
            image = img2.open(str(tmp_path))
            image = ImageTk.PhotoImage(image)
            root.image = image
            imagesprite = updateconvax.create_image(295, 190, image=image, anchor='c')
            time.sleep(1)
            updateconvax.update_idletasks()  # Force redraw
            updateconvax.delete(imagesprite)

            # save the final plot of the trajectories
            if i == finalFrame - 1:
                cv2.imwrite(os.path.join(trajectorydir, 'finalTrajectory.png'), img)
                cv2.imwrite(os.path.join(trajectorydir, 'Plottrajector.png'), mask)
                image = img2.open(str(tmp_path))
                image = ImageTk.PhotoImage(image)
                root.image = image
                imagesprite4 = updateconvax.create_image(263,187, image=image)

        except EOFError:
            continue
        timelapse += Initialtime

    unpacked = zip(frameID, cellIDs, trajectoriesX, trajectoriesY, t)
    with open(os.path.join(csvdir,  'data.csv'), 'wt') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerow(('frameID', 'track_no', 'x', "y","time",))
        for value in unpacked:
            writer.writerow(value)



class Application:
    def __init__(self, master):

        # 1: Create a builder

        self.builder = builder = pygubu.Builder()

        # 2: Load an ui file
        builder.add_from_file('celltracker_html.ui')

        # 3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('mainwindow', master)

        # 4: Get the labeled frame
        self.labelframe1 = builder.get_object("Labelframe_19")

        # 5:  Get the filename or path
        self.pathchooserinput_3 = builder.get_object("pathchooserinput_3")

        # 6: Read the files
        self.button = builder.get_object("Button_10")

        # 7: Create a progress bar
        self.progressdialog = ttk.Progressbar(self.labelframe1, mode='indeterminate', value=0)
        self.progressdialog.grid(row=2, column=0, sticky=N + E + W)

        self.Labelframe_22 = builder.get_object("Labelframe_22")
        self.progressdialog2 = ttk.Progressbar(self.Labelframe_22, mode='indeterminate', value=0)
        self.progressdialog2.grid(row=3, column=0, columnspan=5,sticky=N + E + W)

        # 8:  Segmentation parameters

        self.labelframe2 = builder.get_object("Labelframe_12")

        # 8.1: Scale label
        self.label = Label(self.labelframe2)
        self.label.grid(row=1, column=5, sticky=W)
        self.fixscale = 0.5
        self.label.configure(text=self.fixscale)

        # get the display label
        self.label_10 = self.builder.get_object("Label_10")
        self.label_11 = self.builder.get_object("Label_11")
        self.label_12 = self.builder.get_object("Label_12")
        self.label_13 = self.builder.get_object("Label_13")
        self.label_14 = self.builder.get_object("Label_14")
        self.label_15 = self.builder.get_object("Label_15")
        self.label_16 = self.builder.get_object("Label_16")
        self.label_17 = self.builder.get_object("Label_17")
        self.label_18 = self.builder.get_object("Label_18")
        self.label_19 = self.builder.get_object("Label_19")
        self.label_43 = self.builder.get_object("Label_43")
        self.label_44 = self.builder.get_object("Label_44")
        self.label_45 = self.builder.get_object("Label_45")
        self.label_46 = self.builder.get_object("Label_46")
        self.label_47= self.builder.get_object("Label_47")
        self.label_48 = self.builder.get_object("Label_48")
        self.label_49 = self.builder.get_object("Label_49")
        self.label_50 = self.builder.get_object("Label_50")
        self.label_51 = self.builder.get_object("Label_51")
        self.label_52 = self.builder.get_object("Label_52")
        self.label_63 = self.builder.get_object("Label_63")
        self.label_64 = self.builder.get_object("Label_64")
        self.label_65= self.builder.get_object("Label_65")
        self.label_66 = self.builder.get_object("Label_66")
        self.label_67 = self.builder.get_object("Label_67")
        self.label_68 = self.builder.get_object("Label_68")
        self.label_69 = self.builder.get_object("Label_69")
        self.label_70 = self.builder.get_object("Label_70")
        self.label_71 = self.builder.get_object("Label_71")
        self.label_72 = self.builder.get_object("Label_72")


       # 8.2: Entry
        self.cellEstimate = 200
        self.minDistance = 40

        # 9: Perform segmentation

        self.preview = builder.get_object("Button_1")

        self.convax1 = builder.get_object("Canvas_4")


        self.segmentation = self.builder.get_variable("seg")
        self.color = self.builder.get_variable("background")

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 10: Create a tracking labels

        self.track = builder.get_variable("track")

        self.trackconvax = builder.get_object("Canvas_2")


        #11: Create a file mining

        self.generatefile = builder.get_object("Button_3")

        self.savefile2 = builder.get_object("Button_4")

        #12: about the toolbox

        self.clear = builder.get_object("Button_11")

        builder.connect_callbacks(self)

        ##: set global variable
        self.frames, self.timestamp = [], []

    def readfile_process_on_click(self):
        # Get the path choosed by the user

        path = self.pathchooserinput_3.cget('path')

        # show the path
        if path:
            tkMessageBox.showinfo('You choosed', str(path))

            # Check for the file format
            if '.tif' in path:
                tif = TIFF.open(path, mode='r')

                try:
                    for cc, tframe in enumerate(tif.iter_images()):
                        self.frames.append(tframe)
                        self.progressdialog.step(cc)
                        self.progressdialog.update()


                        # self.mainwindow.update()
                except EOFError or MemoryError:
                    tkMessageBox.showinfo('Error', 'file cant be read!!!')
                    pass
                self.progressdialog.stop()

            if '.avi' or '.gif' or '.mp4' in path:

                cap = cv2.VideoCapture(path)
                cc = 0
                try:
                    while cap.isOpened():
                        ret, img = cap.read()
                        # get the frame in seconds
                        t1 = cap.get(0)
                        self.timestamp.append(t1)
                        if img is None:
                            break
                        self.frames.append(img)
                        self.progressdialog.step(cc)
                        self.progressdialog.update()
                        time.sleep(0.1)
                        cc += 1
                        # self.mainwindow.update()

                except EOFError:
                    tkMessageBox.showinfo('Error', 'file cant be read!!!')
                    pass
                self.progressdialog.stop()
            tmp_img = self.frames[0]
            r = 500.0 / tmp_img.shape[1]
            dim = (500, int(tmp_img.shape[0] * r))

            # perform the actual resizing of the image and show it
            resized = cv2.resize(tmp_img, dim, interpolation=cv2.INTER_AREA)
            mahotas.imsave('raw_image.gif', resized)
            image1 = img2.open('raw_image.gif')


            image1 = ImageTk.PhotoImage(image1)
            root.image1 = image1
            _ = self.convax1.create_image(263, 187, image=image1)

        else:
            tkMessageBox.showinfo("No file", "Choose a file to process")

    # segmentation preview
    def previe_on_click(self):
        "Display the values of the 2 x Entry widget variables"
        self.cellEstimate = self.builder.get_object('Entry_1')
        self.minDistance = self.builder.get_object('Entry_3')
        self.preconvax = self.builder.get_object("Canvas_5")

        self.cellEstimate = self.cellEstimate.get()
        self.minDistance = self.minDistance.get()

        if self.frames:
            # normalize histogram for improving the image contrast
            self.normalizedImage = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
            self.normalizedImage = histogram_equaliz(self.normalizedImage)
            self.segMethod = self.segmentation.get()

            if self.segMethod == 2:
                if self.color.get() == 1:
                    self.prev_image = black_background(self.frames[0], self.kernel)
                    r = 500.0 / self.prev_image.shape[1]
                    dim = (500, int(self.prev_image.shape[0] * r))

                    # perform the actual resizing of the image and show it
                    self.prev_image = cv2.resize(self.prev_image, dim, interpolation=cv2.INTER_AREA)
                    mahotas.imsave('SegImage.gif', self.prev_image)
                    tmp_pre = img2.open('SegImage.gif')
                    tmp_pre = ImageTk.PhotoImage(tmp_pre)
                    root.tmp_pre = tmp_pre
                    segprev = self.preconvax.create_image(263,187, image=tmp_pre)
                if self.color.get() == 2:
                    self.prev_image = white_background(self.frames[0], self.kernel)
                    r = 500.0 / self.prev_image.shape[1]
                    dim = (500, int(self.prev_image.shape[0] * r))

                    # perform the actual resizing of the image and show it
                    self.prev_image = cv2.resize(self.prev_image, dim, interpolation=cv2.INTER_AREA)
                    mahotas.imsave('SegImage.gif', self.prev_image)
                    tmp_pre = img2.open('SegImage.gif')
                    tmp_pre = ImageTk.PhotoImage(tmp_pre)
                    root.tmp_pre = tmp_pre
                    segprev = self.preconvax.create_image(263,187, image=tmp_pre)
            if self.segMethod == 3:
                self.prev_image = harris_corner(self.normalizedImage, int(self.cellEstimate), float(self.fixscale),
                                                int(self.minDistance))
                for corner in self.prev_image:
                    x, y = corner[0]
                    cv2.circle(self.normalizedImage, (int(x),int(y)),5, (0,0,255),-1)

                r = 500.0 / self.normalizedImage.shape[1]
                dim = (500, int(self.normalizedImage.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.normalizedImage = cv2.resize(self.normalizedImage, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.normalizedImage)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(263,187, image=tmp_pre)

            if self.segMethod == 4:
                self.prev_image = shi_tomasi(self.normalizedImage, int(self.cellEstimate), float(self.fixscale),
                                             int(self.minDistance))
                for corner in self.prev_image:
                    x, y = corner[0]
                    cv2.circle(self.normalizedImage, (x, y), 5, (0, 255, 0), -1)

                r = 500.0 / self.normalizedImage.shape[1]
                dim = (500, int(self.normalizedImage.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.normalizedImage = cv2.resize(self.normalizedImage, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.normalizedImage)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(263,187, image=tmp_pre)

            if self.segMethod == 5:
                self.prev_image = basic_seg(self.frames[0], self.frames)
                #print corners
                '''for corner in corners:
                    x, y = corner
                    cv2.circle(self.normalizedImage, (int(x), int(y)), 5, (0, 0, 255), -1)'''



                r = 500.0 / self.prev_image.shape[1]
                dim = (500, int(self.prev_image.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.prev_image = cv2.resize(self.prev_image, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.prev_image)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(263,187, image=tmp_pre)

            if self.segMethod == 1:
                _, self.keypoints = blob_seg(self.frames[0])
                self.im_with_keypoints = cv2.drawKeypoints(self.frames[0], self.keypoints, np.array([]), (0, 0, 255),
                                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



                r = 500.0 / self.im_with_keypoints.shape[1]
                dim = (500, int(self.im_with_keypoints.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.prev_image = cv2.resize(self.im_with_keypoints, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.prev_image)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(263,187, image=tmp_pre)
            return self.segMethod

            if self.segMethod == 10:
                _, cnts = segmentColorImageEGBS(self, self.frames[0])
                self.im_with_keypoints = cv2.drawContours(frame[0], cnts, -1, (255, 0, 0), 1)
                r = 500.0 / self.im_with_keypoints.shape[1]
                dim = (500, int(self.im_with_keypoints.shape[0] * r))

                # perform the actual resizing of the image and show it
                self.prev_image = cv2.resize(self.im_with_keypoints, dim, interpolation=cv2.INTER_AREA)
                mahotas.imsave('SegImage.gif', self.prev_image)
                tmp_pre = img2.open('SegImage.gif')
                tmp_pre = ImageTk.PhotoImage(tmp_pre)
                root.tmp_pre = tmp_pre
                segprev = self.preconvax.create_image(263, 187, image=tmp_pre)

                cv2.imshow("window title", frame[0])
                cv2.waitKey(0)
        else:
            tkMessageBox.showinfo('No file', 'no data is found!!!')



    # perform tracking
    def track_on_click(self):
        if self.frames:
            self.cellEstimate = self.builder.get_object('Entry_1')
            self.minDistance = self.builder.get_object('Entry_3')

            self.timelapse = self.builder.get_object("Entry_2")

            self.timelapse = int(self.timelapse.get())

            self.TrajectoryColor = self.builder.get_variable("2")
            self.ID = self.builder.get_variable("0")

            if self.TrajectoryColor.get() ==1:
                self.Traj = True
            else:
                self.Traj= False

            if self.ID.get() ==1:
                self.CellID  = True
            else:
                self.CellID = False

            if self.frames:
                self.normalizedImage = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
                self.normalizedImage = histogram_equaliz(self.normalizedImage)
            else:
                pass

            if self.segmentation.get() == 2:
                if self.color.get() == 1:
                    self.mask = black_background(self.frames[0], self.kernel)
                    #self.mask = histogram_equaliz(self.mask)
                    self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()),
                                                    float(self.fixscale),
                                                    int(self.minDistance.get()))
                    self.seg = 'watershed'
                if self.color.get() == 2:
                    self.mask = white_background(self.frames[0], self.kernel)
                    self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
                    self.mask = histogram_equaliz(self.mask)

                    self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()),
                                                    float(self.fixscale),
                                                    int(self.minDistance.get()))
                    self.seg = 'watershed'
            if self.segmentation.get() == 3:
                self.initialpoints = harris_corner(self.normalizedImage, int(self.cellEstimate.get()), float(self.fixscale),
                                                   int(self.minDistance.get()))
                self.seg = 'haris'

            if self.segmentation.get() == 4:
                self.initialpoints = shi_tomasi(self.normalizedImage, int(self.cellEstimate.get()), float(self.fixscale),
                                                int(self.minDistance.get()))
                self.seg = 'shi'

            if self.segmentation.get() == 5:
                self.mask =  basic_seg(self.frames[0], self.frames)
                self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()), float(self.fixscale),
                                                int(self.minDistance.get()))
                self.seg = 'basic'

            if self.segmentation.get() == 1:
                self.initialpoints,_ =  blob_seg(self.frames[0])
                self.seg = 'blob'


            # manipulate a tracking method

            if self.track.get() == 8:
                tkMessageBox.showinfo('..','Segmentation method: %s \n' %self.seg, )

                optical_flow(self, self.frames[1:], self.frames[0], self.initialpoints, str(self.seg),
                             int(self.cellEstimate.get()), float(self.fixscale), int(self.minDistance.get()), self.trackconvax, self.progressdialog2, self.timelapse, line=self.Traj, ID=self.CellID, )
            if self.track.get() == 9:
                tkMessageBox.showinfo('..', 'Segmentation method: %s \n' % self.seg, )

                basic_tracking(self, self.frames[1:], self.frames[0], self.initialpoints, str(self.seg),
                             int(self.cellEstimate.get()), float(self.fixscale), int(self.minDistance.get()),
                             self.trackconvax, self.progressdialog2, self.timelapse, line=self.Traj,
                             ID=self.CellID, )

        else:
            tkMessageBox.showinfo('Missing data', 'no data to process')



        def track_on_click(self):
            if self.frames:
                self.cellEstimate = self.builder.get_object('Entry_1')
                self.minDistance = self.builder.get_object('Entry_3')

                self.timelapse = self.builder.get_object("Entry_2")

                self.timelapse = int(self.timelapse.get())

                self.TrajectoryColor = self.builder.get_variable("2")
                self.ID = self.builder.get_variable("0")

                if self.TrajectoryColor.get() == 1:
                    self.Traj = True
                else:
                    self.Traj = False

                if self.ID.get() == 1:
                    self.CellID = True
                else:
                    self.CellID = False

                if self.frames:
                    self.normalizedImage = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
                    self.normalizedImage = histogram_equaliz(self.normalizedImage)
                else:
                    pass

                if self.segmentation.get() == 2:
                    if self.color.get() == 1:
                        self.mask = black_background(self.frames[0], self.kernel)
                        # self.mask = histogram_equaliz(self.mask)
                        self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()),
                                                        float(self.fixscale),
                                                        int(self.minDistance.get()))
                        self.seg = 'watershed'
                    if self.color.get() == 2:
                        self.mask = white_background(self.frames[0], self.kernel)
                        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
                        self.mask = histogram_equaliz(self.mask)

                        self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()),
                                                        float(self.fixscale),
                                                        int(self.minDistance.get()))
                        self.seg = 'watershed'
                if self.segmentation.get() == 3:
                    self.initialpoints = harris_corner(self.normalizedImage, int(self.cellEstimate.get()),
                                                       float(self.fixscale),
                                                       int(self.minDistance.get()))
                    self.seg = 'haris'
                if self.segmentation.get() == 4:
                    self.initialpoints = shi_tomasi(self.normalizedImage, int(self.cellEstimate.get()),
                                                    float(self.fixscale),
                                                    int(self.minDistance.get()))
                    self.seg = 'shi'

                if self.segmentation.get() == 5:
                    self.mask = basic_seg(self.frames[0], self.frames)
                    self.initialpoints = shi_tomasi(self.mask, int(self.cellEstimate.get()), float(self.fixscale),
                                                    int(self.minDistance.get()))

                    self.seg = 'basic'

                if self.segmentation.get() == 6:
                    self.initialpoints,_ = basic_seg(self.frames[0])
                    self.seg = 'blob'


                # manipulate a tracking method

                if self.track.get() == 8:
                    tkMessageBox.showinfo('..', 'Segmentation method: %s \n' % self.seg, )

                    optical_flow(self, self.frames[1:], self.frames[0], self.initialpoints, str(self.seg),
                                 int(self.cellEstimate.get()), float(self.fixscale), int(self.minDistance.get()),
                                 self.trackconvax, self.progressdialog2, self.timelapse, line=self.Traj,
                                 ID=self.CellID, )

                if self.track.get() == 9:
                    tkMessageBox.showinfo('..', 'Segmentation method: %s \n' % self.seg, )

                    optical_flow(self, self.frames[1:], self.frames[0], self.initialpoints, str(self.seg),
                                 int(self.cellEstimate.get()), float(self.fixscale), int(self.minDistance.get()),
                                 self.trackconvax, self.progressdialog2, self.timelapse, line=self.Traj,
                                 ID=self.CellID, )
            else:
                tkMessageBox.showinfo('Missing data', 'no data to process')


    # scale
    def on_scale_click(self, event):

        scale = self.builder.get_object('Scale_1')
        self.fixscale = float("%.1f" % round(scale.get(), 1))
        self.label.configure(text=str(self.fixscale))


    # generate files
    def generate_click(self):
        save_gif = True
        title = ''
        images, imgs = [], []
        for foldername in os.listdir(overlaytrajectorydir):
            images.append(foldername)
        images.sort(key=lambda x: int(x.split('.')[0]))

        for _, file in enumerate(images):
            im = img2.open(os.path.join(overlaytrajectorydir,file))

            imgs.append(im)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

        im_ani = animation.ArtistAnimation(fig, ims, interval=600, repeat_delay=500, blit=False)

        if save_gif:
            im_ani.save(os.path.join(overlaytrajectoryanidir, 'animation.gif'), writer="imagemagick")

        myFormats = [('JPEG / JFIF', '*.jpg'), ('CompuServer GIF', '*.gif'), ]
        filename = asksaveasfilename(filetypes=myFormats)


        if filename:
            im = img2.open(os.path.join(overlaytrajectoryanidir,'animation.gif'))
            original_duration = im.info['duration']
            frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
            frames.reverse()

            ##from images2gif import writeGif
            #print os.path.basename(filename)
            #writeGif('/home/sami/animation.gif', frames, duration=original_duration / 1000.0, dither=0)

    def savefile(self):
        name = asksaveasfilename(initialdir=csvdir)
        f1 = open(os.path.join(name),'wt')
        writer = csv.writer(f1, lineterminator='\n')
        spamReader = csv.reader(open(os.path.join(csvdir,'data.csv')))
        for row in spamReader:
            writer.writerow(row)
        f1.close()


    def save_as_zip(self):
        zf = zipfile.ZipFile("data.zip", "w")
        for dirname, subdirs, files in os.walk(tmppath):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
        zf.close()

        myFormats = [('ZIP files', '*.zip'),  ]
        filenames = asksaveasfilename(initialdir=tmppath,filetypes=myFormats)

        if filenames:
            zf = zipfile.ZipFile(os.path.join(filenames), 'w')

            for dirname, subdirs, files in os.walk(tmppath):
                zf.write(dirname)
                for filename in files:
                    zf.write(os.path.join(str(dirname),filename))
            zf.close()


    def clear_frame(self, master):
        get_ipython().magic('reset -sf')

if __name__ == '__main__':
    root = tk.Tk()
    menu = Menu(root)
    root.config(menu=menu)

    file = Menu(menu)
    file.add_command(label='Exit', command='')

    menu.add_cascade(label='File', menu=file)

    edit = Menu(menu)

    # adds a command to the menu option, calling it exit, and the
    # command it runs on event is client_exit


    # added "file" to our menu
    menu.add_command(label="Open", command='')
    menu.add_command(label="Save", command='')
    menu.add_separator()
    menu.add_cascade(label="Edit", menu=edit)

    # root.geometry('1500x1000')

    #img = Image("photo", file="multimot2.ico")
    ##root.tk.call('wm', 'iconphoto', root._w, img)
    root.title("CellSET:Cell SEgmentation and Tracking")
    app = Application(root)
    root.mainloop()
