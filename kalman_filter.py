import numpy as np
import cv2
from matplotlib import pyplot
import pylab
import scipy.io
import glob
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from numpy.linalg import inv


print "hello world"

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

#get the original detections
x = scipy.io.loadmat('/home/sami/matlabalgorithms/video/raw_fly_detections.mat')
S_frame = 5;
X = x['X']
Y = x['Y']
X = X[0]
Y = Y[0]


# list the number of images
img_list = glob.glob('/home/sami/matlabalgorithms/video/*.jpg')


cap = cv2.VideoCapture('/home/sami/matlabalgorithms/978808.mp4')


im = cv2.imread('/home/sami/Desktop/movies/extractVIDEOS/frame0.jpg', 1)
noOfFrames = 4
m, n, p = im.shape

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


r1 = X[S_frame][:]
r2 = Y[S_frame][:]

r1 = np.hstack(r1)
r2 = np.hstack(r2)
r3 = np.zeros([len(X[S_frame])])
r4 = np.zeros([len(X[S_frame])])
r1 = r1.tolist()
r2 = r2.tolist()
r3 = r3.tolist()
r4 = r4.tolist()
Q  = np.vstack([r1, r2, r3, r4])




Q_estimate = np.empty([4,2000])
Q_estimate[:] = np.NaN

Q_estimate[:,0:len(X[S_frame])] = Q # %estimate of initial location estimation of where the cells are(what we are updating)

Q_loc_estimateY = np.empty([2000, 2000]) # position estimate
Q_loc_estimateY[:] = np.NaN
Q_loc_estimateX = np.empty([2000,2000]) #%  position estimate
Q_loc_estimateX[:] = np.NaN
P_estimate = P;  #%covariance estimator
strk_trks = np.zeros((1,2000)) #%counter of how many strikes a track has gotten
nD = len(X[S_frame]) #%initize number of detections

#Find indicies that you need to replace
where_are_NaNs = np.isnan(Q_estimate)
#Replace the NaNs with 1
nf = Q_estimate[where_are_NaNs]=1

nF = 29
# get the algorithms

# for each frame
t = 4
for i in range(269,536):

    frame = cv2.imread('/home/sami/matlabalgorithms/video/image%d.jpg'%i, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # make the given detection matrix
    xt = np.trunc(X[t])
    yt = np.trunc(Y[t])
    Q_loc_meas = np.hstack([xt, yt])



    ## do the kalman filter
    # predict next state of the cell with the last state and predicted motion
    nD = len(X[t])
    for f in range(0,29):
        Q_estimate[:,f] = np.dot(A, Q_estimate[:,f]) + np.dot(B,u)

    # predict the next covariance

    ATranspose =  A.T  # transpose for alignment

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
    est_dist = est_dist[0:nF, nF + 0:] # limit to just the   tracks   to  detection   distances
    assign, cost = linear_sum_assignment(est_dist)

    #assign = np.vstack(assign)

    # ignore far detection
    rejectDetection = np.ones(29, dtype=int)

    for f in range(0,29):
        if assign[f] >= 0:
            value = int(est_dist[f, assign[f]])
            if est_dist[f,assign[f]] < 50:
                rej = 1
                rejectDetection[f]
                rejectDetection[f] = rej
        else:
            rejectDetection[f] = 0

    assign = assign * rejectDetection


    # now lets apply the assignment to the update
    k = 0
    for a in range(len(assign)):
        if assign[a] >= 0:
            XX = np.dot(K,(Q_loc_meas[assign[a],:].T - np.dot(C, Q_estimate[:, k])))
            Q_estimate[:,k] = Q_estimate[:,k] + XX


        k += 1
    t +=1

    # update covariance estimation.
    P = np.dot((np.eye(4) - np.dot(K, C)), P)


    # Store    data
    Q_loc_estimateX[t, 0:nF] = Q_estimate[1, 0:nF]
    Q_loc_estimateY[t, 0:nF] = Q_estimate[2, 0:nF]

    print Q_loc_estimateX[t, 0:nF]
