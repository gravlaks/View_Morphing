import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from HW.sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    U, D, VT = np.linalg.svd(E)
    V = VT.T
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    Z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0],
    ])

    Q1 = U @ W @ V.T
    Q2 = U @ W.T @ V.T
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2
    u3 = U[:,-1]

    RT = np.zeros((4,3,4))
    RT[0, :, :3], RT[0, :, 3] = R1, +u3
    RT[1, :, :3], RT[1, :, 3] = R1, -u3
    RT[2, :, :3], RT[2, :, 3] = R2, +u3
    RT[3, :, :3], RT[3, :, 3] = R2, -u3

    return RT
    raise Exception('Not Implemented Error')

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    m = image_points.shape[0]

    A = np.zeros((2*m,4))
    for i in range(m):
        A[2*i+0,:] = image_points[i,0] * camera_matrices[i, 2, :] - camera_matrices[i, 0, :]
        A[2*i+1,:] = image_points[i,1] * camera_matrices[i, 2, :] - camera_matrices[i, 1, :] 
    
    _, _, VT = np.linalg.svd(A)
    point_3d = VT[-1,:3] / VT[-1,3]

    return point_3d
    raise Exception('Not Implemented Error')

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    m = image_points.shape[0]

    hp3d = np.ones((4,))
    hp3d[:3] = point_3d

    e = np.zeros((2*m,))
    for i in range(m):
        M0 = camera_matrices[i,0,:]; M0P = M0.dot(hp3d)
        M1 = camera_matrices[i,1,:]; M1P = M1.dot(hp3d)
        M2 = camera_matrices[i,2,:]; M2P = M2.dot(hp3d)
        e[2*i+0] = M0P / M2P - image_points[i, 0]
        e[2*i+1] = M1P / M2P - image_points[i, 1]
    
    return e
    raise Exception('Not Implemented Error')

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    m = camera_matrices.shape[0]

    hp3d = np.ones((4,))
    hp3d[:3] = point_3d

    J = np.zeros((2*m,4))
    for i in range(m):
        M0 = camera_matrices[i,0,:]; M0P = M0.dot(hp3d)
        M1 = camera_matrices[i,1,:]; M1P = M1.dot(hp3d)
        M2 = camera_matrices[i,2,:]; M2P = M2.dot(hp3d)
        J[2*i+0, :] = (M2P * M0 - M0P * M2) / np.square(M2P)
        J[2*i+1, :] = (M2P * M1 - M1P * M2) / np.square(M2P)

    return J[:, :3]
    raise Exception('Not Implemented Error')

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    point_3d = linear_estimate_3d_point(image_points, camera_matrices)

    for _ in range(20):
        e = reprojection_error(point_3d, image_points, camera_matrices)
        J = jacobian(point_3d, camera_matrices)
        point_3d -= np.linalg.inv(J.T @ J) @ J.T @ e
    
    return point_3d

    raise Exception('Not Implemented Error')

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    n, m, _ = image_points.shape
    RT0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    RT = estimate_initial_RT(E) # Hx3x4
    H, _, _ = RT.shape

    camera_matrices = np.zeros((H,m,3,4))
    camera_matrices[:, 0, :, :] = (K @ RT0).reshape((1,3,4))
    camera_matrices[:, 1, :, :] = np.einsum('ij,hjl->hil', K, RT)

    tally = np.zeros((4,))
    for h in range(H):
        for i in range(n):
            c0_point_3d = nonlinear_estimate_3d_point(image_points[i,:,:], camera_matrices[h,:,:,:])
            c1_point_3d = RT[h, :, :3].dot(c0_point_3d) + RT[h, :, 3]
            tally[h] += (c0_point_3d[2] > 0) and (c1_point_3d[2] > 0)
    
    k = np.argmax(tally)

    return RT[k]
    raise Exception('Not Implemented Error')

#if __name__ == '__main__':
    #run_pipeline = True

    ## Load the data
    #image_data_dir = 'data/statue/'
    #unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    #unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    #image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        #sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    #focal_length = 719.5459
    #matches_subset = np.load(os.path.join(image_data_dir,
        #'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    #dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               #allow_pickle=True, encoding='latin1')
    #fundamental_matrices = np.load(os.path.join(image_data_dir,
        #'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    ## Part A: Computing the 4 initial R,T transformations from Essential Matrix
    #print('-' * 80)
    #print("Part A: Check your matrices against the example R,T")
    #print('-' * 80)
    #K = np.eye(3)
    #K[0,0] = K[1,1] = focal_length
    #E = K.T.dot(fundamental_matrices[0]).dot(K)
    #im0 = imread(image_paths[0])
    #im_height, im_width, _ = im0.shape
    #example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        #[0.1019, 0.9948, 0.0045, -0.0089],
        #[0.2041, -0.0254, 0.9786, 0.0331]])
    #print("Example RT:\n", example_RT)
    #estimated_RT = estimate_initial_RT(E)
    #print('')
    #print("Estimated RT:\n", estimated_RT)

    ## Part B: Determining the best linear estimate of a 3D point
    #print('-' * 80)
    #print('Part B: Check that the difference from expected point ')
    #print('is near zero')
    #print('-' * 80)
    #camera_matrices = np.zeros((2, 3, 4))
    #camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    #camera_matrices[1, :, :] = K.dot(example_RT)
    #unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    #estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        #camera_matrices.copy())
    #expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    #print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    ## Part C: Calculating the reprojection error and its Jacobian
    #print('-' * 80)
    #print('Part C: Check that the difference from expected error/Jacobian ')
    #print('is near zero')
    #print('-' * 80)
    #estimated_error = reprojection_error(
            #expected_3d_point, unit_test_matches, camera_matrices)
    #estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    #expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    #print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    #expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         #[0., 154.33943931, 36.51165089],
         #[141.87950588, -14.27738422, -56.20341644],
         #[21.9792766, 149.50628901, 32.23425643]])
    #print("Jacobian Difference: ", np.fabs(estimated_jacobian
        #- expected_jacobian).sum())

    ## Part D: Determining the best nonlinear estimate of a 3D point
    #print('-' * 80)
    #print('Part D: Check that the reprojection error from nonlinear method')
    #print('is lower than linear method')
    #print('-' * 80)
    #estimated_3d_point_linear = linear_estimate_3d_point(
        #unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    #estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        #unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    #error_linear = reprojection_error(
        #estimated_3d_point_linear, unit_test_image_matches,
        #unit_test_camera_matrix)
    #print("Linear method error:", np.linalg.norm(error_linear))
    #error_nonlinear = reprojection_error(
        #estimated_3d_point_nonlinear, unit_test_image_matches,
        #unit_test_camera_matrix)
    #print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    ## Part E: Determining the correct R, T from Essential Matrix
    #print('-' * 80)
    #print("Part E: Check your matrix against the example R,T")
    #print('-' * 80)
    #estimated_RT = estimate_RT_from_E(E,
        #np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    #print("Example RT:\n", example_RT)
    #print('')
    #print("Estimated RT:\n", estimated_RT)

    ## Part F: Run the entire Structure from Motion pipeline
    #if not run_pipeline:
        #sys.exit()
    #print('-' * 80)
    #print('Part F: Run the entire SFM pipeline')
    #print('-' * 80)
    #frames = [0] * (len(image_paths) - 1)
    #for i in range(len(image_paths)-1):
        #frames[i] = Frame(matches_subset[i].T, focal_length,
                #fundamental_matrices[i], im_width, im_height)
        #bundle_adjustment(frames[i])
    #merged_frame = merge_all_frames(frames)

    ## Construct the dense matching
    #camera_matrices = np.zeros((2,3,4))
    #dense_structure = np.zeros((0,3))
    #for i in range(len(frames)-1):
        #matches = dense_matches[i]
        #camera_matrices[0,:,:] = merged_frame.K.dot(
            #merged_frame.motion[i,:,:])
        #camera_matrices[1,:,:] = merged_frame.K.dot(
                #merged_frame.motion[i+1,:,:])
        #points_3d = np.zeros((matches.shape[1], 3))
        #use_point = np.array([True]*matches.shape[1])
        #for j in range(matches.shape[1]):
            #points_3d[j,:] = nonlinear_estimate_3d_point(
                #matches[:,j].reshape((2,2)), camera_matrices)
        #dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    #fig = plt.figure(figsize=(10,10))
    #ax = fig.gca(projection='3d')
    #ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        #c='k', depthshade=True, s=2)
    #ax.set_xlim(-5, 5)
    #ax.set_ylim(-5, 5)
    #ax.set_zlim(0, 10)
    #ax.view_init(-100, 90)

    #plt.show()
