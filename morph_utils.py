from time import sleep
import numpy as np
import cv2 as cv
import cvxpy as cp
import dlib
from imutils import face_utils
from HW import p2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

def homogenize(pt):
    """
        Take in x ~ (N, 2)
        Return  y ~ (N, 3) (an additional column of ones)
    """
    k = pt.shape[0]
    pt2 = np.ones((k+1,))
    pt2[:k] = pt
    return pt2

def load_mona_lisas():
    """
        Load mona lisa, flip it and return noth images + the dimensions
    """
    # generate left and right facing mona lisa
    mona_1 = cv.imread("data/torstein/left.jpg")
    dims = (mona_1.shape[1], mona_1.shape[0])
    print("dims", dims)
    

    mona_2 = cv.imread("data/torstein/right.jpg")


    scale_percent = 30 # percent of original size
    width = int(dims[0] * scale_percent / 100)
    height = int(dims[1]* scale_percent / 100)
    dims = (width, height)
    print()
    # resize image
    mona_1 = cv.resize(mona_1, dims, interpolation = cv.INTER_AREA)
    mona_2 = cv.resize(mona_2, dims, interpolation = cv.INTER_AREA)
    F = np.eye(3)
    F[0, 0] = -1
    F[0, 2] = dims[0]
    #mona_2 = cv.warpPerspective(mona_1, F, dims)

    return mona_1, mona_2, dims

def find_face(im):
    """
        Take an image containing 1 face and return the pixel location of 68 facial features
    """
    rects = detector(im, 1)
    for rect in rects:
        landmarks = predictor(im, rect)
        pts = face_utils.shape_to_np(landmarks)
        pts = np.array(list(map(homogenize, pts)))
        break

    return pts

def rotvec(u, theta):
    """
        Find the 3x3 rotatino matrix which rotates a vector theta radians around the point u
    """
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - np.cos(theta)
    x = u[0]
    y = u[1]
    return np.array([[t*x*x + c, t*x*y, s*y],
                    [t*x*y, t*y*y + c, -s*x],
                    [-s*y, s*x, c]])

def rot_z(theta):
    """
        Rotation matrix about the z axis
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ +c, -s, 0 ],
        [ +s, +c, 0 ],
        [ 0, 0, 1 ],
    ])

def get_fundamental(pts_1, pts_2):
    """
        Fun wrapper !!
    """
    F, _ = cv.findFundamentalMat(pts_1, pts_2)
    return F


def get_fundamental_calib(pts_1, pts_2):
    """
        Fun wrapper !!
    """
    K = np.load("data/calibration/K.pkl.npy")

    ## Undistort 
    E, _ = cv.findEssentialMat(pts_1[:, :2], pts_2[:, :2], K, method=cv.RANSAC)
    F = np.linalg.inv(K.T)@E@np.linalg.inv(K)
    return F

def get_homographies(F):
    """
        Implement the homographies method presented in the original view morph paper
    """
    e_1 = p2.compute_epipole(F.T)
    e_1 /= np.linalg.norm(e_1)
    e_2 = p2.compute_epipole(F)
    e_2 /= np.linalg.norm(e_2)

    d_1 = np.array([ -e_1[1], e_1[0], 0 ])
    X = F @ d_1
    d_2 = np.array([ -X[1], X[0], 0 ])

    theta_1 = np.arctan(e_1[2]/(d_1[1] * e_1[0] - d_1[0] * e_1[1]))
    theta_2 = np.arctan(e_2[2]/(d_2[1] * e_2[0] - d_2[0] * e_2[1]))
    phi_1 = - np.arctan(e_1[1]/e_1[0])
    phi_2 = - np.arctan(e_2[1]/e_2[0])

    R_1t = rotvec(d_1, theta_1)
    R_2t = rotvec(d_2, theta_2)
    R_1p = rot_z(phi_1)
    R_2p = rot_z(phi_2)

    F_h = R_1p @ R_1t @ F @ R_2t.T @ R_2p.T

    T = np.array([
        [1, 0, 0],
        [0, -F_h[1,2], -F_h[2,2]],
        [0, 0, F_h[2,1]],
    ])

    H_1 = R_1p @ R_1t
    H_2 = R_2p @ R_2t

    return H_1, H_2

def apply_perspective(H, x):
    """
        H ~ (3, 3)
        x ~ (n, 3)
        ->
        y ~ (n, 3)
    """
    y = np.einsum('ij,lj->li', H, x)
    return y / y[:,2].reshape((-1, 1))

def get_framed_homographies(F, f_1, f_2, dims):
    """
        Same as the homographies matrix except the images are guaranteed to not blow up in size or clip
    """
    H_1, H_2 = get_homographies(F)

    f = np.vstack([
        apply_perspective(np.linalg.inv(H_1), f_1),
        apply_perspective(np.linalg.inv(H_2), f_2),
    ])[:, :2]

    x_min = np.min(f[:,0]) / (2 * dims[0])
    x_max = np.max(f[:,0]) / (2 * dims[0])
    y_min = np.min(f[:,1]) / (2 * dims[1])
    y_max = np.max(f[:,1]) / (2 * dims[1])

    dims = (
        2 * dims[0],
        2 * dims[1]
    )

    T = np.diag([1/(x_max-x_min), 1/(y_max-y_min), 1])
    T[0,2] = -x_min
    T[1,2] = -y_min

    T = np.linalg.inv(T)

    return H_1 @ T, H_2 @ T, dims

