import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from HW.epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # TODO: Implement this method!
    n, _ = points_im1.shape

    x = np.zeros((2,n,2))
    x[0,:,:] = points_im1[:,:2]
    x[1,:,:] = points_im2[:,:2]

    x_hat = x - np.einsum('ij,k->ikj', np.mean(x, axis=1), np.ones(n))
    D = x_hat.transpose([0, 2, 1]).reshape((4, n))

    u, s, vh = np.linalg.svd(D)
    u = u[:, :3]
    s = s[:3]
    vh = vh[:3, :]
    S = np.diag(np.sqrt(s)) @ vh
    M = u @ np.diag(np.sqrt(s))
    return S, M

    raise Exception('Not Implemented Error')

#if __name__ == '__main__':
#    for im_set in ['data/set1', 'data/set1_subset']:
#        # Read in the data
#        im1 = imread(im_set+'/image1.jpg')
#        im2 = imread(im_set+'/image2.jpg')
#        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
#        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
#        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
#        assert (points_im1.shape == points_im2.shape)
#
#        # Run the Factorization Method
#        structure, motion = factorization_method(points_im1, points_im2)
#
#        # Plot the structure
#        fig = plt.figure()
#        ax = fig.add_subplot(121, projection = '3d')
#        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
#        ax.set_title('Factorization Method')
#        ax = fig.add_subplot(122, projection = '3d')
#        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
#        ax.set_title('Ground Truth')
#
#        plt.show()
#