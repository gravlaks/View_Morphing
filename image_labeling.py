import cv2


coordinates = []
coordinates2 = []

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks

    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        i = ''
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), (x, y), font,
                    0.4, (255, 0, 0), 1)
        cv2.circle(img, [x,y], 2, (0, 0, 255))
        coordinates.append((str(x) + ' ' + str(y)))
        cv2.imshow('image', img)

def click_event2(event, x, y, flags, params):
    # checking for left mouse clicks

    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        i = ''
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2, str(i), (x, y), font,
                    0.4, (255, 0, 0), 1)
        cv2.circle(img2, [x,y], 2, (0, 0, 255))
        coordinates2.append((str(x) + ' ' + str(y)))
        cv2.imshow('image2', img2)


# driver function
if __name__ == "__main__":
    # reading the image
    img = cv2.imread('data/h_h.jpg', 1)
    img2 = cv2.imread('data/h_v.jpg', 1)

    # displaying the image


    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.namedWindow("image2", cv2.WINDOW_NORMAL)

    screen_res = 2160, 1440
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    #resized window width and height
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    #cv2.resizeWindow('image', window_width, window_height)
    #cv2.resizeWindow('image2', window_width, window_height)


    cv2.imshow('image', img)

    cv2.imshow('image2', img2)

    cv2.setMouseCallback('image', click_event)
    cv2.setMouseCallback('image2', click_event2)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    with open('coordinates.txt', 'w') as f:
        for coord in coordinates:
            f.write(coord)
            f.write('\n')

    with open('coordinates2.txt', 'w') as f:
        for coord in coordinates2:
            f.write(coord)
            f.write('\n')

    # close the window
    cv2.destroyAllWindows()