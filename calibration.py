import argparse
import cv2
from marker_dictionary import marker_dictionary
import numpy as np
import glob, os

print("[INFO] loading image...")
os.chdir("./data/calibration")
filenames = glob.glob("*.jpg")

#filenames = os.listdir("./data/calibration")
print(filenames)
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()

print("Dictionary set")
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
print("images read", len(images))
image_points = []
object_points = []

for img in images:
    print("...")
    points2d, points3d = [], []
    
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
	parameters=arucoParams)
    print(len(corners))
    if ids is  None or corners is None:
        
        continue
    for id, corner in zip(ids, corners):
        bl = corner[0, 2].astype(int)
        print(bl)

        points2d.append(bl)
        points3d.append(get_3d_from(id.item()))
        img = cv2.circle(img, [bl[0], bl[1]], 2, 0x444444, 2)
        img = cv2.putText(img, str(id.item()), [bl[0]-3, bl[1]-3], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0x444444)
    if len(points2d)>=4:
        image_points.append(np.array(points2d).astype(np.float32))
        object_points.append(np.array(points3d).astype(np.float32))
    
print("done fetching", len(image_points))
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    object_points, image_points, img.shape[:2][::-1], None, None)


# Displaying required output
print(" Camera matrix:")
print(matrix)
np.save("K.pkl", matrix)
print("\n Distortion coefficient:")
print(distortion)
np.save("dist.pkl", distortion)
print("\n Rotation Vectors:")
print(r_vecs)
  
print("\n Translation Vectors:")
print(t_vecs)
