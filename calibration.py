import argparse
import cv2


print("[INFO] loading image...")
filename = "data/test1.jpg"
image = cv2.imread(filename)

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
	parameters=arucoParams)


for id, corner in zip(ids, corners):
    bl = corner[0, 2].astype(int)
    print(bl)
    image = cv2.circle(image, [bl[0], bl[1]], 2, 0x444444, 2)

cv2.imshow("image", image)
cv2.waitKey(0)
