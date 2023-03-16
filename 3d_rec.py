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
    point_3d = linear_estimate_3d_point(image_points, camera_matrices)

    for _ in range(10):
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

def get_normal_map_from_tris(P, tris):
    """
        P    ~ (N, 4) - 3d points
        tris ~ (M, 3) - triangle indices
    """
    normal_map = dict()
    for tri in tris:
        a, b, c = np.split(P[tri], [1, 2])
        a = a[0, :3]
        b = b[0, :3]
        c = c[0, :3]
        ijk = tuple(tri)
        v = np.cross(a-b, c-b)
        v /= np.linalg.norm(v)
        normal_map[ijk] = v
    return normal_map


def generate_manual(s = 0.5, scale = 1, save = True, subdivide = True):
    # load images
    I_1, f_1, n_1, I_2, f_2, n_2, dim = load_manual(scale)
    f_1 = homogenize_array(f_1)
    f_2 = homogenize_array(f_2)
    #f_1 = find_face(I_1)
    #f_2 = find_face(I_2)

    f_count = len(f_1)


    # find triangulation
    delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

    F = np.zeros((f_count, 2, 2))
    F[:, 0, :] = f_1[:,:2]
    F[:, 1, :] = f_2[:,:2]

    # attempt 3d reconstruction of face
    E, K = get_calibration(f_1, f_2)
    RT = estimate_RT_from_E(E, F, K)

    epilines_1 = np.array([E @ f_2[0], E @ f_2[1]])
    epilines_2 = np.array([E.T @ f_1[0], E.T @ f_1[1]])

    v_1 = np.cross(epilines_1[0], epilines_1[1])
    v_2 = np.cross(epilines_2[0], epilines_2[1])
    v_1 = v_1 / np.linalg.norm(v_1)
    v_2 = v_2 / np.linalg.norm(v_2)


    M_1 = K @ np.array([
        [ 1, 0, 0, 0, ],
        [ 0, 1, 0, 0, ],
        [ 0, 0, 1, 0, ],
    ])

    M_2 = K @ RT

    M = np.zeros((2, 3, 4))
    M[0, :, :] = M_1
    M[1, :, :] = M_2

    P = np.ones((f_count, 4))
    for k in range(f_count):
        P[k,:3] = nonlinear_estimate_3d_point(F[k], M)

    p_1 = apply_projection(M_1, P)[:, :2]
    p_2 = apply_projection(M_2, P)[:, :2]
    M_s = s * M_1 + (1 - s) * M_2
    f_s = apply_projection(M_s, P)[:, :2]

    show_faces = False
    if show_faces == True:
        geo_utils.plot_tris_2d(p_1, delu)
        plt.show()

        geo_utils.plot_tris_2d(p_2, delu)
        plt.show()

        geo_utils.plot_tris_2d(f_s, delu)
        plt.show()
    
    show_triang = False

    if show_triang:
        geo_utils.plot_tris_3d(P[:,:3], delu)
        plt.show()
    
    if subdivide:
        for i in range(1):
            Q = geo_utils.catmull_clark(P, delu)
            P = np.ones((Q.shape[0], 4))
            P[:, :3] = Q[:, :3]
            f_1 = apply_projection(M_1, P)
            f_2 = apply_projection(M_2, P)
            f_s = apply_projection(M_s, P)
            delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

        if show_triang:
            geo_utils.plot_tris_3d(P[:,:3], delu)
            plt.show()

    normal_map = get_normal_map_from_tris(P, delu)

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

        if normal_map[tuple(t_i)] @ v_2 <= 0.5: #then triangle can't be seen from camera 1
            #print(normal_map[tuple(t_i)] @ v_2)
            #mix = cv.warpAffine(I_2.copy(), S_2, dim)
            mix = cv.addWeighted(
                cv.warpAffine(I_1.copy(), S_1, dim), 0.1,
                cv.warpAffine(I_2.copy(), S_2, dim), 0.9,
                0
            )
        elif normal_map[tuple(t_i)] @ v_1 <= 0.5:
            #print("hello")
            #mix = cv.warpAffine(I_1.copy(), S_1, dim)
            mix = cv.addWeighted(
                cv.warpAffine(I_1.copy(), S_1, dim), 0.9,
                cv.warpAffine(I_2.copy(), S_2, dim), 0.1,
                0
            )
        else:
        # mix the two images on the correct location
            mix = cv.addWeighted(
                cv.warpAffine(I_1.copy(), S_1, dim), s,
                cv.warpAffine(I_2.copy(), S_2, dim), 1 - s,
                0
            )
        '''
        mix = cv.addWeighted(
            cv.warpAffine(I_1.copy(), S_1, dim), s,
            cv.warpAffine(I_2.copy(), S_2, dim), 1 - s,
            0
        )
        '''
        I_s |= local_mask & mix
    
    if save:
        cv.imwrite(f'output/occlusion_0.5_newmix3thresh_{s:.2f}.jpg', I_s)

    return I_s.copy()

if __name__ == "__main__":
    I = generate_manual(s=0.5, scale=0.4, save=True, subdivide=True)
    rgb = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()