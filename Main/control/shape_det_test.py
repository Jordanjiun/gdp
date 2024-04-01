# import libraries
import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)


# Empty function; just a placeholder cv2.createTrackbar
def empty(a):
    pass


# Window for trackbars
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)

# Trackbars to adjust thresholding
cv2.createTrackbar("Threshold1", "Parameters", 89, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
# Trackbar to adjust area limit of geometrical detection
cv2.createTrackbar("Area", "Parameters", 1500, 10000, empty)


# Function for stiching all windows into a single window
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# Function for geometrical detection using contours
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")

        if area > areaMin:

            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Create bounding box around detected shape
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y,), (x + w, y + h), (0, 255, 0), 5)
            # Insert text next to bounding box
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

            # if number of contours detected = 4, rectangle is detected, otherwise no rectangle is detected
            if len(approx) == 4:
                print("rectangle")
            else:
                print("no rectangle")


# Loop over frames from the video stream
while True:
    check, img = cam.read()
    imgContour = img.copy()
    # Blur the image
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    # Grayscale the image
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    # Apply thresholding based on the value of trackbar applied
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    # Reduce/filter noise of image
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour)

    # Stack windows together in one single window base on the stackImages function
    imgStack = stackImages(0.8, ([img, imgGray, imgCanny],
                                 [imgDil, imgContour, imgContour]))

    cv2.imshow('camera', imgStack)

    # If esc key is pressed, break from loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# end the code
cam.release()
cv2.destroyAllWindows()
