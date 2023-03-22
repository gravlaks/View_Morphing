import numpy as np
import matplotlib.pyplot as plt
from HW.p1 import *
from HW.epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # TODO: Implement this method!
    _, _, vh = np.linalg.svd(F.T)
    return vh[-1,:] / vh[-1,-1]
    raise Exception('Not Implemented Error')

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    # TODO: Implement this method!
    H, W = im.shape

    T = np.array([
        [1, 0, - W / 2],
        [0, 1, - H / 2],
        [0, 0, 1]
    ])

    e1, e2, _ = T @ e

    al = 2. * (e1 >= 0).astype(np.float64) - 1
    E = al / np.sqrt(e1**2 + e2**2)
    R = np.array([
        [E * e1, E * e2, 0],
        [E * -e2, E * e1, 0],
        [0, 0, 1],
    ])

    f, _, _ = R @ T @ e

    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1/f, 0, 1],
    ])

    return np.linalg.inv(T) @ G @ R @ T
    raise Exception('Not Implemented Error')

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    H2 = compute_H(e2, im2)

    C = np.array([
        [0, -e2[2], +e2[1]],
        [+e2[2], 0, -e2[0]],
        [-e2[1], +e2[0], 0],
    ])

    M = C @ F + np.outer(e2, np.ones((3,)))

    p1 = (H2 @ M @ points1[:,:].T).T
    p1 /= np.outer(p1[:,2], np.ones(3))
    p2 = (H2 @ points2[:,:].T).T
    p2 /= np.outer(p2[:,2], np.ones(3))
    W = p1[:,:] # ~ (37, 3)
    b = p2[:,0] # ~ (37,)
    #a = np.linalg.pinv(W) @ b
    a = np.linalg.lstsq(W, b, rcond=None)[0]

    Ha = np.array([
        [a[0], a[1], a[2]],
        [0, 1, 0],
        [0, 0, 1],
    ])

    H1 = Ha @ H2 @ M

    return H1, H2
    raise Exception('Not Implemented Error')

#if __name__ == '__main__':
#    # Read in the data
#    im_set = 'data/set1'
#    im1 = imread(im_set+'/image1.jpg')
#    im2 = imread(im_set+'/image2.jpg')
#    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
#    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
#    assert (points1.shape == points2.shape)
#
#    F = normalized_eight_point_alg(points1, points2)
#    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
#    e1 = compute_epipole(F.T)
#    e2 = compute_epipole(F)
#    print("e1", e1)
#    print("e2", e2)
#
#    # Find the homographies needed to rectify the pair of images
#    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
#    print('')
#
#    # Transforming the images by the homographies
#    new_points1 = H1.dot(points1.T)
#    new_points2 = H2.dot(points2.T)
#    new_points1 /= new_points1[2,:]
#    new_points2 /= new_points2[2,:]
#    new_points1 = new_points1.T
#    new_points2 = new_points2.T
#    rectified_im1, offset1 = compute_rectified_image(im1, H1)
#    rectified_im2, offset2 = compute_rectified_image(im2, H2)
#    new_points1 -= offset1 + (0,)
#    new_points2 -= offset2 + (0,)
#
#    # Plotting the image
#    F_new = normalized_eight_point_alg(new_points1, new_points2)
#    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
#    plt.show()
#