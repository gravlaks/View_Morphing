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
W = 1024

class Semaphore:
    def __init__(self):
        self.value = 0

def click_event(event, x, y, flags, params):
    I, clicked, color, size, coordinates = params

    if event == cv2.EVENT_RBUTTONDOWN:
        coordinates += [ (-1, -1) ]
        clicked.value = True
    if event == cv2.EVENT_LBUTTONDOWN:
        I = cv2.circle(I, [x,y], size, color, -1)
        coordinates += [ (x, y) ]
        clicked.value = True

def main(images):
    I = { image : cv2.imread(image, 1) for image in images }
    coordinates = { image : [] for image in images }

    # displaying the image
    finished_labeling = False
    clicked = Semaphore()

    while True:
        color = random_color()

        for image in images:
            cv2.namedWindow(image, cv2.WINDOW_NORMAL)
            cv2.imshow(image, I[image])
            cv2.moveWindow(image, 100, 100)
            cv2.resizeWindow(image, W, W)
            cv2.setMouseCallback(image, click_event, (I[image], clicked, color, size, coordinates[image]))

            while not clicked.value:
                key = cv2.waitKey(10)
                if key != -1:
                    cv2.destroyAllWindows()
                    max_count = np.min([len(coordinates[image]) for image in images])
                    return { image : coordinates[image][:max_count] for image in images }

            cv2.destroyAllWindows()
            clicked.value = False


# driver function
if __name__ == "__main__":
    # reading the image
    images = [
        'data/torstein/left.jpg',
        'data/torstein/front.jpg',
        'data/torstein/right.jpg',
    ]

    data = main(images)

    if data:
        json.dump(data, open('data/manual_multiple.json', 'w'))