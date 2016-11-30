
import cv2
import numpy as np
from libtiff import TIFF
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage import data
from skimage import img_as_ubyte
from skimage import morphology





path  = '/home/sami/Essen/ctr_multipage (1).tif'

tif = TIFF.open(path, mode='r')
for cc, tframe in enumerate(tif.iter_images()):
    #self.frames.append(tframe)
    im = cv2.cvtColor(tframe, cv2.COLOR_BGR2GRAY)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 20
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 15

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
    keypoints = detector.detect(tframe)
    print keypoints[0].pt
    print len(keypoints)
    pts = np.float([keypoints[idx].pt for idx in range(len(keypoints))]).reshape(-1, 1, 2)
    print pts
    exit()
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(tframe, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
