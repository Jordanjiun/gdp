# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import sys


# include camera matrix and distortion coefficients from camera calibration + marker sizes
matrix_coefficients = np.array([[966.90841671, 0, 611.58804072],
                                [0, 965.65159698, 371.44866427],
                                [0, 0, 1]])
distortion_coefficients = np.array([[6.86566894e-2, -1.79345595, 2.17383609e-3, -1.84014871e-3, 1.17946050e+1]])
marker_size = 4  # centimeters

# fps counter initialisation
prev_frame_time = 0
new_frame_time = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
              "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
              "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
              "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
              "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
              "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
              "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
              "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
              "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
              "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
              "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}

# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)


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


# Function for geometrical detection using contours
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")

        if area > areaMin:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # if number of contours detected = 4, rectangle is detected, otherwise no rectangle is detected
            if len(approx) == 4:
                # Create bounding box around detected shape
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgDil, (x, y,), (x + w, y + h), (255, 255, 255), 5)

                # Insert text next to bounding box
                cv2.putText(imgDil, f"Rectangle", (x + w + 20, y + 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(imgDil, f"Rectangle Detected", (7, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3, cv2.LINE_AA)
            else:
                pass


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 1000 pixels
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1000)

    if not ret:
        break

    # code for shape detection
    img = frame.copy()
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

    # code for marker detection
    new_frame_time = time.time()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect ArUco markers in the input frame
    marker_corners, marker_IDs, reject = cv2.aruco.detectMarkers(gray_frame,
                                                                 arucoDict,
                                                                 parameters=arucoParams)

    # loop over the detected ArUCo corners
    if marker_corners:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners,
                                                            marker_size,
                                                            matrix_coefficients,
                                                            distortion_coefficients)
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculating the distance
            distance = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)

            # Draw the pose of the marker
            point = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec[i], tvec[i], 6, 6)
            cv2.putText(frame, f"id: {ids} Dist: {round(distance, 2)}", top_right,
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"x:{round(tvec[i][0][0], 1)} y: {round(tvec[i][0][1], 1)} ", bottom_right,
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # fps calculations
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(frame, f"FPS:{fps}", (7, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)

    # resize and show the output frames
    resized_img = cv2.resize(imgDil, (600, 450))
    imgc = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    resized_frame = cv2.resize(frame, (600, 450))
    hori = np.concatenate((imgc, resized_frame), axis=1)
    cv2.imshow("Test", hori)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
