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

# Variables
cX_array = []
cY_array = []
new_cY_threshold = []
length = 0
ori_done_flag = False
x_done_flag = False
y_done_flag = False
alignment_done = False

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
print(arucoDict)
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 1000 pixels
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1000)
    if not ret:
        break

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
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Locations for centroid distance calculation
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Calculating the distance
            distance = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)
            # distance = (tvec[i][0][2])

            # Draw the pose of the marker
            # point = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec[i], tvec[i], 6, 6)
            # Text for distance of markers
            cv2.putText(frame, f"id: {ids} Dist: {round(distance, 2)}", top_right,
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            # Text for x & y coordinates
            cv2.putText(frame, f"x:{round(tvec[i][0][0], 1)} y: {round(tvec[i][0][1], 1)} ", bottom_right,
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            cX_array.append(cX)
            cY_array.append(cY)

            # Orientation Process
            reference1 = 444  # Reference tag for orientation
            reference2 = 222  # Reference tag for orientation
            ref_222_y = 100  #
            ref_444_y = 0  #

            if ori_done_flag == False:
                for i in range(marker_IDs.size):
                    if marker_IDs[i][0] == reference1:
                        ref_444_y = tvec[i][0][1]
                        # print("444: ", ref_444_y)
                    elif marker_IDs[i][0] == reference2:
                        ref_222_y = tvec[i][0][1]
                        # print("222: ", ref_222_y)

                # Threshold to account for external forces; wind, etc
                threshold_ori = 0.2  # Threshold value; tentative
                if ref_444_y <= ref_222_y + threshold_ori and ref_444_y >= ref_222_y - threshold_ori:
                    print(ref_444_y, ref_222_y)
                    print("Stop Ori")
                    ori_done_flag = True
                else:
                    print("Rotate")

            # Alignment Process
            reference = 444  # Reference tag for alignment

            # Alignment in the x-axis
            # Flag for sequence control
            if ori_done_flag == True:
                if x_done_flag == False and alignment_done == False:
                    for i in range(marker_IDs.size):
                        if reference == marker_IDs[i][0]:
                            if tvec[i][0][0] > 1.0:
                                print("Left")
                            elif tvec[i][0][0] < -1.0:
                                print("right")
                            else:
                                print("x stop")
                                x_done_flag = True

                # Alignement in the y-axis
                if x_done_flag == True and alignment_done == False:
                    for i in range(marker_IDs.size):
                        if reference == marker_IDs[i][0]:
                            if tvec[i][0][1] >= -5.0:
                                print("Down")
                            elif tvec[i][0][1] <= -6.0:
                                print("Up")
                            else:
                                print("y stop")
                                y_done_flag = True
                                alignment_done = True

        if y_done_flag == True:
            # Height Control Process
            new_cY = cY_array[0] - 50  # Threshold value; tentative
            for number in range(101):
                new_cY_threshold.append(new_cY + number)

            # Tag rearrangement
            for tagNumber in range(len(cX_array)):
                if tagNumber is not 0:
                    for number in range(len(new_cY_threshold)):
                        if new_cY_threshold[number] == cY_array[tagNumber]:
                            if cX_array[0] > cX_array[tagNumber]:
                                length = cX_array[0] - cX_array[tagNumber]
                            else:
                                length = cX_array[tagNumber] - cX_array[0]
                            start_pt = (cX_array[0], cY_array[0])
                            end_pt = (cX_array[tagNumber], cY_array[tagNumber])
                            cv2.line(frame, start_pt, end_pt, (0, 0, 255), 2)
                            print("Length", length)

                            if length < 350:
                                print("Going Down")
                            else:
                                print("STOP")
                                y_done_flag = False

            cX_array.clear()
            cY_array.clear()
            new_cY_threshold.clear()
            length = 0

    # fps calculations
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(frame, f"FPS:{fps}", (7, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
