import argparse
import cv2
from marker_dictionary import marker_dictionary
import numpy as np


print("[INFO] loading image...")
filenames = ["data/test1.jpg", "data/test2.jpg"]

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()


def get_3d_from(id):

    ## Units in m, cube_dim includes white border
    x, y = marker_dictionary[id]
    cube_dim = 0.005 
    white_dim = 0.001
    Z = 0
    X = x*cube_dim+white_dim
    Y = y*cube_dim+white_dim

    return [X, Y, Z]

images = [cv2.imread(fname) for fname in filenames]
image_points = []
object_points = []

for img in images:
    points2d, points3d = [], []
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
	parameters=arucoParams)
    for id, corner in zip(ids, corners):
        bl = corner[0, 2].astype(int)
        print(bl)

        points2d.append(bl)
        points3d.append(get_3d_from(id.item()))
        img = cv2.circle(img, [bl[0], bl[1]], 2, 0x444444, 2)
        img = cv2.putText(img, str(id.item()), [bl[0]-3, bl[1]-3], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0x444444)
    image_points.append(points2d)
    object_points.append(points3d)
image_points = np.array(image_points).astype(np.float32)
object_points = np.array(object_points).astype(np.float32)
print(object_points)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    object_points, image_points, img.shape[:2][::-1], None, None)




# Displaying required output
print(" Camera matrix:")
print(matrix)
  
print("\n Distortion coefficient:")
print(distortion)
  
print("\n Rotation Vectors:")
print(r_vecs)
  
print("\n Translation Vectors:")
print(t_vecs)
