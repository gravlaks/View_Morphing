import cv2 as cv

img = cv.imread('data/triangle_.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv.threshold(img,127,255,0)

contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
print(cnt)

(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv.circle(img,center,radius,(0,255,0),2)

cv.imshow("Delaunay", img)
cv.waitKey(0)