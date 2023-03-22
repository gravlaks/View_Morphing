import cv2
import json
import numpy as np

"""
    !!! READ 
    This program will alternate between two images (image_1, image_2).
    You will select a shared keypoint (left click) or a keypoint not viewable from the the other image (right click)
    You may quit the program by pressing any key while viewing the first image.
    The labels will be saved to data/manual.json
"""

random_color = lambda: tuple(map(int, 255 * np.random.random(3)))
clicked = False
color = random_color()
size = 20
yshared_coordinates_1 = []
nshared_coordinates_1 = []
yshared_coordinates_2 = []
nshared_coordinates_2 = []
W = 1024

def click_event_1(event, x, y, flags, params):
    global img_1, clicked, color, size, nshared_coordinates_1, yshared_coordinates_1

    if event == cv2.EVENT_RBUTTONDOWN:
        img_1 = cv2.circle(img_1, [x,y], size, (255, 255, 255), -1)
        nshared_coordinates_1 += [ (x, y) ]
        clicked = True
    if event == cv2.EVENT_LBUTTONDOWN:
        img_1 = cv2.circle(img_1, [x,y], size, color, -1)
        yshared_coordinates_1 += [ (x, y) ]
        clicked = True

def click_event_2(event, x, y, flags, params):
    global img_2, clicked, color, size, nshared_coordinates_2, yshared_coordinates_2

    if event == cv2.EVENT_RBUTTONDOWN:
        img_2 = cv2.circle(img_2, [x,y], size, (255, 255, 255), -1)
        nshared_coordinates_2 += [ (x, y) ]
        clicked = True
    if event == cv2.EVENT_LBUTTONDOWN:
        img_2 = cv2.circle(img_2, [x,y], size, color, -1)
        yshared_coordinates_2 += [ (x, y) ]
        clicked = True

# driver function
if __name__ == "__main__":
    # reading the image
    image_1 = 'data/torstein/left.jpg'
    image_2 = 'data/torstein/right.jpg'
    filename="data/manual2.json"
    img_1 = cv2.imread(image_1, 1)
    img_2 = cv2.imread(image_2, 1)

    # displaying the image
    finished_labeling = False

    while True:
        color = random_color()
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img_1)
        cv2.moveWindow('image', 100, 100)
        cv2.resizeWindow('image', W, W)
        cv2.setMouseCallback('image', click_event_1)

        while not clicked:
            key = cv2.waitKey(10)
            if key != -1:
                finished_labeling = True
                break

        if finished_labeling:
            break

        cv2.destroyAllWindows()
        clicked = False

        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.imshow('image2', img_2)
        cv2.moveWindow('image2', 100, 100)
        cv2.resizeWindow('image2', W, W)
        cv2.setMouseCallback('image2', click_event_2)
        while not clicked:
            cv2.waitKey(10)
        cv2.destroyAllWindows()
        clicked = False
    cv2.destroyAllWindows()

    print(f"Saving labeling with {len(yshared_coordinates_1)} shared keys and {len(nshared_coordinates_1)} non-shared keys to {filename}")

    data = [
        {
            "path" : image_1,
            "shared_keys" : yshared_coordinates_1,
            "non_shared_keys" : nshared_coordinates_1,
        },
        {
            "path" : image_2,
            "shared_keys" : yshared_coordinates_2,
            "non_shared_keys" : nshared_coordinates_2,
        }
    ]
    json.dump(data, open(filename, 'w'))