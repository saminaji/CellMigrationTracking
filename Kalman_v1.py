import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pylab
import scipy.io
import glob
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from numpy.linalg import inv
from sklearn.neighbors import NearestNeighbors
from libtiff import TIFF

# get the segmentation algorithm
def shi_tomasi(image, maxCorner=200, qualityLevel=0.5, MinDistance=20):
    # detect corners in the image

    corners = cv2.goodFeaturesToTrack(image,
                                      maxCorner,
                                      qualityLevel,
                                      MinDistance,
                                      mask=None,
                                      blockSize=7)

    return corners


def readtiff(path):
    frames = []
    tif = TIFF.open(path, mode='r')

    try:
        for cc, tframe in enumerate(tif.iter_images()):
            frames.append(tframe)
            if cc > 200:
                break
    except EOFError:
        pass
    return frames

#get the original detections

path = '/home/sami/Downloads/ctr_multipage (2).tif'
#path = '/home/sami/9K5F98PS_F00000011.avi'

frames = readtiff(path)
#frames,_ =filereader(path)

old_frame = frames[0]

# params for ShiTomasi corner detection

feature_params = dict(maxCorners=50,
                      qualityLevel = 0.05,
                      minDistance = 5,
                      blockSize = 7 )
# Parameters for lucas kanade optical flow

lk_params = dict(winSize=(20, 20), maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5))

  # Create some random colors

color = np.random.randint(0, 255, (200, 3))

# Take first frame and find corners in it

old_gray_image = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)




p0 = cv2.goodFeaturesToTrack(old_gray_image, mask=None, **feature_params)



# list the number of images
img_list = frames
im = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
noOfFrames = 4
m, n, p = frames[0].shape

# Kalman filter parameters
dt = 1.0 # Sampling rate

# Prediction matrix
A = np.array([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
# Control matrix
B = np.array([dt ** 2 / 2, dt ** 2 / 2, dt, dt])

# %this is our measurement function C, that we apply to the state estimate Q to get our expect next/new measurement
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# define the acceleration
u = 0

# Acceleration magnitude
HexAccel_noise_mag = 1; #process noise: the variability in how
tkn_x = 0.1  # acceleration in the horizontal direction
tkn_y = 0.1  # acceleration in the vertical direction


Ez = [[tkn_x,0], [0, tkn_y]]

#Ex convert the process noise (stdv) into covariance matrix
#P = Ex; % estimate of initial cells position variance (covariance matrix)
Ex = np.dot([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],[ 0, dt ** 4 / 4, 0, dt ** 3 / 2],
     [dt ** 3 / 2, 0, dt ** 2, 0],[ 0, dt ** 3 / 2, 0,dt ** 2]], HexAccel_noise_mag ** 2)

P = Ex


#%% initize result variables
Q_loc_meas = []; #% the cell detecions  extracted by the detection algo

#%% initize estimation variables for two dimensions
X1, Y1 = [], []
for _,row in enumerate(p0):
    C1, D1 = row.ravel()
    X1.append(C1)
    Y1.append(D1)
r1 = X1
r2 = Y1

r1 = np.hstack(r1)
r2 = np.hstack(r2)
r3 = np.zeros([len(p0)])
r4 = np.zeros([len(p0)])
r1 = r1.tolist()
r2 = r2.tolist()
r3 = r3.tolist()
r4 = r4.tolist()
Q  = np.vstack([r1, r2, r3, r4])

Q_estimate = np.empty([4,2000])
Q_estimate[:] = np.NaN

Q_estimate[:,0:len(p0)] = Q # %estimate of initial location estimation of where the cells are(what we are updating)

Q_loc_estimateY = np.empty([2000, 2000]) # position estimate
Q_loc_estimateY[:] = np.NaN
Q_loc_estimateX = np.empty([2000,2000]) #%  position estimate
Q_loc_estimateX[:] = np.NaN
P_estimate = P;  #%covariance estimator
strk_trks = np.zeros((1,2000)) #%counter of how many strikes a track has gotten
nD = len(p0) #%initize number of detections
color = np.random.randint(0, 255, (1000, 3))
#Find indicies that you need to replace
where_are_NaNs = np.isnan(Q_estimate)
#Replace the NaNs with 1
nf = Q_estimate[where_are_NaNs]=1

nF = len(p0)

# for each frame
t = 4
idx = 0

firstDetections, updatedTrackIdx, updateDetections = [], [], []
for i, frame in enumerate(frames[1:]):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1 = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)


    X2, Y2 = [], []
    for _, row in enumerate(p1):
        C2, D2 = row.ravel()
        X2.append(C2)
        Y2.append(D2)

    # make the given detection matrix
    xt = np.vstack(X2)
    yt = np.vstack(Y2)

    Q_loc_meas = np.hstack([xt, yt])


    ## do the kalman filter
    # predict next state of the cell with the last state and predicted motion
    nD = len(p0)
    for f in range(0,nD):
        Q_estimate[:,f] = np.dot(A, Q_estimate[:,f]) + np.dot(B,u)

    # predict the next covariance

    ATranspose = A.T  # transpose for alignment
    firstMat = np.dot(A, P)
    secondMat = np.dot(firstMat, ATranspose)
    P = secondMat + Ex


    # compute Kalman gain

    CTranspose  = C.T # tranpose the matrix for alignment
    leftSide = np.dot(P, CTranspose)

    rightSideA = np.dot(C,P)
    rightSideB = np.dot(rightSideA,CTranspose)
    rightSideC = rightSideB + Ez
    rightSide = inv(rightSideC)

    K = np.dot(leftSide,rightSide)

    ## now  we assign the detections to  estimated  track   positions
    # make    the     distance(cost)     matrice    between    all    pairs    rows = tracks, coln =    % detections

    xx = np.vstack([Q_estimate[0:2,0:nF].T, Q_loc_meas])

    est_dist = pdist(xx)
    est_dist = squareform(est_dist)
    est_dist = est_dist[0:nF, nF:] # limit to just the   tracks   to  detection   distances
    assign, cost = linear_sum_assignment(est_dist)

    # ignore far detection
    rejectDetection = np.ones(nF, dtype=int)


    for f in range(0,assign.shape[0]):

        if assign[f] >= 0:
            if est_dist[f,assign[f]] < 50:
                rej = 1
                rejectDetection[f] = rej
            else:
                rejectDetection[f] = 0


        assign = assign * rejectDetection


    # now lets apply the assignment to the update
    k = 0
    for a in range(len(assign)):
        if IndexError:
            continue
        if assign[a] >= 0:
            XX = np.dot(K,(Q_loc_meas[assign[a],:].T - np.dot(C, Q_estimate[:, k])))
            Q_estimate[:,k] = Q_estimate[:,k] + XX


        k += 1


    # update covariance estimation.
    P = np.dot((np.eye(4) - np.dot(K, C)), P)


    # Store data
    Q_loc_estimateX[i, 0:nF] = Q_estimate[0, 0:nF]
    Q_loc_estimateY[i, 0:nF] = Q_estimate[1, 0:nF]


    # ok, now  that  we  have   our assignments and updates, lets  find   the   new  detections and
    # lost   trackings find   the  new  detections.basically, anything  that  doesnt get assigned is a
    #new  tracking
    new_trk = [];
    logicalIndices = []
    indices = Q_loc_meas.shape[0]



    for ind in range(0,Q_loc_meas.shape[0]):
        if ind in assign:
            logicalIndices.append(ind)
        else:
            logicalIndices.append(1)

    #new_trk = cmp(Q_loc_meas,assign)
    ##exit()
    new_trk = Q_loc_meas[logicalIndices].T


    #if new_trk.any() :
    #    print [new_trk, np.zeros(2, new_trk.shape[0])]

    #    Q_estimate[:, nF + 0:nF + len(new_trk)] =  [new_trk, np.zeros(2, new_trk.shape[1])]
    #    nF = nF + len(new_trk) # number  of track   estimates    with new ones included


    # give  a strike to any tracking  that   didn't get matched up to a detection

    no_trk_list = np.where(assign ==[])[0]
    if no_trk_list:
        strk_trks[no_trk_list] = strk_trks(no_trk_list) + 1


    # if a track has a strike greater than 6, delete the tracking.i.e.  % make  it nan first

    bad_trks = np.where(strk_trks > 6)[0]
    Q_estimate[:, bad_trks] = np.NaN


    prevPoints = np.hstack([X1, Y1])

    if i == 1:
        for ii , row in enumerate(prevPoints):
            g, d = row.ravel()
            firstDetections.append(np.float32([g, d]))
            updateDetections.append(np.float32([g, d]))
            updatedTrackIdx.append(idx)
            idx += 1

    firstDetections = np.vstack(firstDetections)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(firstDetections)

    T = Q_loc_estimateX.shape[1];
    Ms = [3, 5]; #% marker

    t = 5
    nextPoints = []
    for Dc in range(0,nF):
        if Q_loc_estimateX[t, Dc]:
            Sz = Dc % 2 + 1 #% pick
            Cz = Dc % 6 + 1 #% pick        color
            if t < 21:
                st = t - 1
            else:
                st = 19
            # get the estimated next location (state)
            tmX = Q_loc_estimateX[t - st:t, Dc]
            tmY = Q_loc_estimateY[t - st:t, Dc]
            nextPoints.append([tmX[3],tmY[3]])

    nextPoints = np.vstack(nextPoints)
    secondDetections = updateDetections

    if i > 269:
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(updateDetections)
        print np.vstack(updateDetections)
        print nextPoints
        updatedTrackIdx = []
        for tt, row in enumerate(nextPoints):
            r2 = row
            z, y = r2.ravel()
            test = [z, y]

            # find the closet point to the testing  in the training data
            nearestpoint = neigh.kneighbors(test)


            trackID = int(nearestpoint[1][0])
            # if trackID:
            updatedTrackIdx.append(trackID)
            distance = float(nearestpoint[0][0])
            updateDetections[trackID] = test

    mask = np.zeros_like(frame,)

    for ii, new in enumerate(nextPoints):

        cellId = updatedTrackIdx[ii]

        a, b = np.float32(new.ravel())

        c, d = np.float32(secondDetections[cellId])

        mask = cv2.line(mask, (a, b), (c, d), (0,255,0), 2)

        mask = cv2.putText(mask, "%d "% cellId, (int(c), int(d)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                            (0,255,0))

        img = cv2.add(frame, mask)

        edged2 = np.hstack([mask, img])
        cv2.imshow('track', edged2)
        cv2.imshow('masked', mask)
        cv2.waitKey(33)

   



