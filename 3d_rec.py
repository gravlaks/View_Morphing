import cv2
import numpy as np
from morph_utils import*
import scipy.spatial
import utils
from tqdm import tqdm
from itertools import count
import geo_utils
import matplotlib.pyplot as plt

def apply_projection(A, x):
    """
        in : 
            A ~ (3, 4) - Projection matrix
            x ~ (N, 4) - homogenous 3d points
        out :
            y ~ (N, 3) - homogenous 2d points
    """
    y = np.einsum('ij,nj->ni', A, x)
    return y / y[:, 2].reshape((-1, 1))

def estimate_3d_linear(M_1, p_1, M_2, p_2):
    """
        in :
            M_1 ~ (3, 4) - projection matrix of image 1
            p_1 ~ (N, 3) - homogenous 2d points from image 1
            M_2 ~ (3, 4) - projection matrix of image 2
            p_2 ~ (N, 3) - homogenous 2d points from image 2
        out :
            P   ~ (N, 4) - homogenous 3d points
    """
    N = p_1.shape[0]
    A = np.zeros((4, 4))
    P = np.zeros((N, 4))

    for k, p, q in zip(count(0), p_1, p_2):
        A[:, 0] = M_1[0,:] - p[0] * M_1[2,:]
        A[:, 1] = M_1[1,:] - p[1] * M_1[2,:]
        A[:, 2] = M_2[0,:] - q[0] * M_2[2,:]
        A[:, 3] = M_2[1,:] - q[1] * M_2[2,:]

        _, _, VT = np.linalg.svd(A)
        P[k, :] = VT[-1, :] / VT[-1, 3]

    return P

def reprojection_error(P, M_1, p_1, M_2, p_2):
    """
        in :
            P   ~ (N, 4) - homogenous 3d points
            M_1 ~ (3, 4) - projection matrix of image 1
            p_1 ~ (N, 3) - homogenous 2d points from image 1
            M_2 ~ (3, 4) - projection matrix of image 2
            p_2 ~ (N, 3) - homogenous 2d points from image 2
        out :
            e   ~ (N,4) - L2 error vector ([[e1x_1,e1y_1,e2x_1,e2y_1], ..., [e1x_n,e1y_n,e2x_n,e2y_n])
    """
    N = P.shape[0]
    e = np.zeros((N,4))
    e[:,:2] = (p_1 - apply_projection(M_1, P))[:,:2]
    e[:,2:] = (p_2 - apply_projection(M_2, P))[:,:2]
    return e

def jacobian(P, M_1, p_1, M_2, p_2):
    """
        in :
            P   ~ (N, 4) - homogenous 3d points
            M_1 ~ (3, 4) - projection matrix of image 1
            p_1 ~ (N, 3) - homogenous 2d points from image 1
            M_2 ~ (3, 4) - projection matrix of image 2
            p_2 ~ (N, 3) - homogenous 2d points from image 2
        out :
            J   ~ (N, 4, 3) - jacobian
    """
    N = P.shape[0]
    J = np.zeros((N,4,4))

    for k, Q in enumerate(P):
        J[k,0,:] = ((M_1[0,:].T @ Q) * M_1[2,:] - (M_1[2,:].T @ Q) * M_1[0,:]) / (M_1[2,:].T @ Q)**2
        J[k,1,:] = ((M_1[1,:].T @ Q) * M_1[2,:] - (M_1[2,:].T @ Q) * M_1[1,:]) / (M_1[2,:].T @ Q)**2
        J[k,2,:] = ((M_2[0,:].T @ Q) * M_2[2,:] - (M_2[2,:].T @ Q) * M_2[0,:]) / (M_2[2,:].T @ Q)**2
        J[k,3,:] = ((M_2[1,:].T @ Q) * M_2[2,:] - (M_2[2,:].T @ Q) * M_2[1,:]) / (M_2[2,:].T @ Q)**2
    
    return J[:,:,:3]

def estimate_3d_nonlinear(M_1, p_1, M_2, p_2):
    """
        in :
            P_0 ~ (N, 4) - initial guess of homogenous 3d points
            M_1 ~ (3, 4) - projection matrix of image 1
            p_1 ~ (N, 3) - homogenous 2d points from image 1
            M_2 ~ (3, 4) - projection matrix of image 2
            p_2 ~ (N, 3) - homogenous 2d points from image 2
        out :
            P   ~ (N, 4) - homogenous 3d points
    """
    P = estimate_3d_linear(M_1, p_1, M_2, p_2)
    N = P.shape[0]

    for _ in range(50):
        e = reprojection_error(P, M_1, p_1, M_2, p_2)
        for n, J in enumerate(jacobian(P, M_1, p_1, M_2, p_2)):
            P[n, :3] = -0.01*np.linalg.inv(J.T @ J) @ J.T @ e[n, :]

    return P

def get_projection_matrices(p_1, p_2, E, K):
    """
        in :
            p_1 ~ (N, 3) - homogenous 2d points from image 1
            p_2 ~ (N, 3) - homogenous 2d points from image 2
            E   ~ (N, 3) - The essential matrix of images 1, 2
            K   ~ (N, 3) - The instrinsics of camera 1, 2
        out :
            M_1 ~ (3, 4) - projection matrix of image 1
            M_2 ~ (3, 4) - projection matrix of image 2
    """
    W = np.array([
        [ 0, -1, 0],
        [ 1, 0, 0],
        [ 0, 0, 1],
    ])

    Z = np.array([
        [ 0, 1, 0],
        [ -1, 0, 0],
        [ 0, 0, 1],
    ])

    U, _, VT = np.linalg.svd(E)
    t_x = U @ Z @ U.T
    t_1 = t_x @ np.array([ 0, 0, +1 ])
    t_2 = t_x @ np.array([ 0, 0, -1 ])
    R_1 = U @ W @ VT
    R_1 = np.linalg.det(R_1) * R_1
    R_2 = U @ W.T @ VT
    R_2 = np.linalg.det(R_2) * R_2

    M_def = np.zeros((3, 4))
    M_def[:3,:3] = np.eye(3)
    M_def = K @ M_def

    T_list = [
        np.vstack([R_1.T, t_1.T]).T,
        np.vstack([R_1.T, t_2.T]).T,
        np.vstack([R_2.T, t_1.T]).T,
        np.vstack([R_2.T, t_2.T]).T,
    ]

    tally = np.zeros(4)
    for k, T_alt in enumerate(T_list):
        M_alt = K @ T_alt
        P_1 = estimate_3d_nonlinear(M_def, p_1, M_alt, p_2)
        T = np.zeros((4,4))
        T[:3, :4] = T_alt
        T[3,3] = 1
        P_2 = np.einsum('ij,nj->ni', T, P_1)
        tally[k] = np.sum((P_1[:,2] > 0) & (P_2[:,2] > 0))
    
    index = np.argmax(tally)
    M_alt = K @ T_list[index]

    return M_def, M_alt

def generate_manual(s = 0.5, scale = 1, save = True):
    # load images
    I_1, f_1, n_1, I_2, f_2, n_2, dim = load_manual(scale = scale)
    f_1 = homogenize_array(f_1)
    f_2 = homogenize_array(f_2)

    f_count = len(f_1)

    # find triangulation
    delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

    # attempt 3d reconstruction of face
    E, K = get_calibration(f_1, f_2)
    M_1, M_2 = get_projection_matrices(f_1, f_2, E, K)
    P = estimate_3d_nonlinear(M_1, f_1, M_2, f_2)
    geo_utils.plot_tris_3d(P, delu)
    plt.show() # fail :((

    f_s = s * f_1 + (1 - s) * f_2

    # Find the coordiantes for all triangles
    tset_1 = f_1[delu][:, :, :2]
    tset_2 = f_2[delu][:, :, :2]
    tset_s = f_s[delu][:, :, :2]

    I_s = 0 * I_1.copy()
    local_mask = 0 * I_1.copy()

    for t_1, t_2, t_s, t_i in zip(tset_1, tset_2, tset_s, delu):
        # find the affine transformations which send the triangles in each image to the composed location
        S_1 = cv.getAffineTransform(t_1.astype(np.float32), t_s.astype(np.float32))
        S_2 = cv.getAffineTransform(t_2.astype(np.float32), t_s.astype(np.float32))

        # find the mask for the triangle
        local_mask = cv.drawContours(0 * local_mask.copy(), [t_s.astype(np.int64)], -1, (255, 255, 255), -1)

        # mix the two images on the correct location
        mix = cv.addWeighted(
            cv.warpAffine(I_1.copy(), S_1, dim), s,
            cv.warpAffine(I_2.copy(), S_2, dim), 1 - s,
            0
        )

        I_s |= local_mask & mix
    
    if save:
        cv.imwrite(f'output/manual{s:.2f}.jpg', I_s)

    return I_s.copy()

import utils

if __name__ == '__main__':
    make_gif = False

    if make_gif:
        frames = []

        for s in tqdm(np.linspace(0.1, 0.9, 11)):
            frames += [ generate_manual(s, 0.2, save = False) ]

        frames += frames[::-1]
    
        utils.create_gif('output/manual.gif', frames)
    else:
        frame = generate_manual(0.5, scale = 0.2, save = False)
        cv.imshow('image', frame)
        cv.waitKey(0)