from zmqRemoteApi import RemoteAPIClient
import numpy as np
import argparse
import imutils
import cv2
import sys

print('Program started')

# ---------- ArUco initiliasation ----------

# include camera matrix and distortion coefficients from camera calibration + marker sizes
matrix_coefficients = np.array([[887.4260997, 0, 510.73233271],
                                [0, 887.18140881, 511.64810058],
                                [0, 0, 1]])
distortion_coefficients = np.array([[2.77182258e-02, -7.96486124e-01, -7.68838494e-05,
                                     -4.19556557e-04, 4.99171203e+00]])
marker_size = 2  # centimeters
frame_width = 500  # for opencv

# camera coord fudge values
xadj = 22
yadj = -22

# variables for control
cx_array = []
cy_array = []
new_cy_threshold = []

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
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()


# ---------- Define functions ----------


# function to detect markers, any other values apart from 0 to activate control
def detect_markers(control=0):
    global x_done_flag, maxi, dumb, alignment_done, threshold_ori,\
        ref_444_y, ref_222_y, y_done_flag, reference1, ori_done_flag, cam_orient_new, xadj, yadj, everything_done
    img, resx, resy = sim.getVisionSensorCharImage(cam)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resy, resx, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    frame = imutils.resize(img, width=frame_width)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect ArUco markers in the input frame
    marker_corners, marker_ids, reject = cv2.aruco.detectMarkers(gray_frame,
                                                                 arucoDict,
                                                                 parameters=arucoParams)
    # loop over the detected ArUCo corners
    if marker_corners:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners,
                                                            marker_size,
                                                            matrix_coefficients,
                                                            distortion_coefficients)
        total_markers = range(0, marker_ids.size)
        for ids, corners, i in zip(marker_ids, marker_corners, total_markers):
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            top_right = corners[0].ravel()
            bottom_right = corners[2].ravel()

            # Calculating the distance
            distance = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)

            # Draw the pose of the marker
            cv2.putText(frame, f"id: {ids} Dist: {round(distance, 2)}", top_right,
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"x:{round(tvec[i][0][0] + xadj, 1)} y: {round(-tvec[i][0][1] + yadj, 1)} ",
                        bottom_right,
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cx = int((topLeft[0] + bottomRight[0]) / 2.0)
            cy = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cx_array.append(cx)
            cy_array.append(cy)

            if control == 0:
                sim.setObjectOrientation(cam, floor, cam_orient)
            else:
                if not everything_done:
                    # Orientation Process
                    reference1 = 444  # Reference tag for orientation
                    reference2 = 222  # Reference tag for orientation

                    cv2.putText(frame, f"Control Sequence Initiated:", (7, 35), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2, cv2.LINE_AA)

                    # draw crosshair
                    cv2.line(frame, (250, 150), (250, 350), (0, 0, 255), 2)
                    cv2.line(frame, (150, 250), (350, 250), (0, 0, 255), 2)

                    # initiate max
                    if dumb == 0:
                        for j in range(marker_ids.size):
                            if marker_ids[j][0] == reference1:
                                ref_444_y = tvec[j][0][1]
                            elif marker_ids[j][0] == reference2:
                                ref_222_y = tvec[j][0][1]
                                maxi = ref_222_y - ref_444_y
                            dumb = dumb + 1

                    # rotation
                    if not ori_done_flag:
                        for j in range(marker_ids.size):
                            if marker_ids[j][0] == reference1:
                                ref_444_y = tvec[j][0][1]
                            elif marker_ids[j][0] == reference2:
                                ref_222_y = tvec[j][0][1]

                        threshold_ori = 0.4  # Threshold value; tentative
                        if ref_222_y + threshold_ori >= ref_444_y >= ref_222_y - threshold_ori:
                            cam_orient_new = sim.getObjectOrientation(cam, -1)
                            ori_done_flag = True
                        elif ref_222_y + threshold_ori >= ref_444_y:
                            cv2.putText(frame, f"Drone Rotating Left", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                        2, (0, 255, 0), 2, cv2.LINE_AA)
                            rot[2] = rot[2] + (rotate * (abs((ref_222_y - ref_444_y) / maxi)))
                            sim.setObjectOrientation(target, floor, rot)
                        elif ref_222_y - threshold_ori <= ref_444_y:
                            cv2.putText(frame, f"Drone Rotating Right", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                        2, (0, 255, 0), 2, cv2.LINE_AA)
                            rot[2] = rot[2] - (rotate * (abs((ref_222_y - ref_444_y) / maxi)))
                            sim.setObjectOrientation(target, floor, rot)
                        else:
                            pass

                    # Alignment in the x-axis
                    # Flag for sequence control
                    if ori_done_flag:
                        sim.setObjectOrientation(cam, floor, cam_orient_new)
                        for j in range(marker_ids.size):
                            if marker_ids[j][0] == reference1:
                                ref_444_y = tvec[j][0][1]
                            elif marker_ids[j][0] == reference2:
                                ref_222_y = tvec[j][0][1]
                        if ref_222_y + threshold_ori >= ref_444_y >= ref_222_y - threshold_ori:
                            pass
                        else:
                            ori_done_flag = False
                        if x_done_flag == False and alignment_done == False:
                            for j in range(marker_ids.size):
                                if reference1 == marker_ids[j][0]:
                                    if round((tvec[j][0][0] + xadj), 1) > 0.5:
                                        cv2.putText(frame, f"Drone Moving Right", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 255, 0), 2, cv2.LINE_AA)
                                        pos[0] = pos[0] + move
                                        sim.setObjectPosition(target, floor, pos)
                                    elif round((tvec[j][0][0] + xadj), 1) < -0.5:
                                        cv2.putText(frame, f"Drone Moving Left", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 255, 0), 2, cv2.LINE_AA)
                                        pos[0] = pos[0] - move
                                        sim.setObjectPosition(target, floor, pos)
                                    else:
                                        x_done_flag = True

                        # Alignement in the y-axis
                        if x_done_flag == True and alignment_done == False:
                            sim.setObjectOrientation(cam, floor, cam_orient_new)
                            for j in range(marker_ids.size):
                                if reference1 == marker_ids[j][0]:
                                    if round((tvec[j][0][0] + xadj), 1) > 0.5 or round((tvec[j][0][0] + xadj), 1) < -0.5:
                                        x_done_flag = False
                                        alignment_done = False
                                    else:
                                        pass
                            for k in range(marker_ids.size):
                                if reference1 == marker_ids[k][0]:
                                    if round(-tvec[k][0][1] + yadj) > 0.5:
                                        cv2.putText(frame, f"Drone Moving Forward", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 255, 0), 2, cv2.LINE_AA)
                                        pos[1] = pos[1] + move
                                        sim.setObjectPosition(target, floor, pos)
                                    elif round(-tvec[k][0][1] + yadj) < -0.5:
                                        cv2.putText(frame, f"Drone Moving Backward", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 255, 0), 2, cv2.LINE_AA)
                                        pos[1] = pos[1] - move
                                        sim.setObjectPosition(target, floor, pos)
                                    else:
                                        y_done_flag = True
                                        alignment_done = True

                    if y_done_flag:
                        sim.setObjectOrientation(cam, floor, cam_orient_new)
                        for j in range(marker_ids.size):
                            if marker_ids[j][0] == reference1:
                                ref_444_y = tvec[j][0][1]
                            elif marker_ids[j][0] == reference2:
                                ref_222_y = tvec[j][0][1]
                        if ref_222_y + threshold_ori >= ref_444_y >= ref_222_y - threshold_ori:
                            pass
                        else:
                            ori_done_flag = False

                        for k in range(marker_ids.size):
                            if reference1 == marker_ids[k][0]:
                                if round((tvec[k][0][0] + xadj), 1) > 0.5 or round((tvec[k][0][0] + xadj), 1) < -0.5:
                                    x_done_flag = False
                                    y_done_flag = False
                                    alignment_done = False
                                elif round(-tvec[k][0][1] + yadj) > 0.5 or round(-tvec[k][0][1] + yadj) < -0.5:
                                    y_done_flag = False
                                    alignment_done = False
                                else:
                                    pass

                        # Height Control Process
                        for m in range(marker_ids.size):
                            if marker_ids[m][0] == reference1:
                                height = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)
                                # very poor height adj fudging
                                if height >= 35:
                                    cv2.putText(frame, f"Drone Descending", (7, 100), cv2.FONT_HERSHEY_PLAIN,
                                                2, (0, 255, 0), 2, cv2.LINE_AA)
                                    pos[2] = pos[2] - descend
                                    sim.setObjectPosition(target, floor, pos)
                                    xadj = xadj - 0.0134
                                    yadj = yadj + 0.0134
                                else:
                                    everything_done = True
                            else:
                                pass
                else:
                    cv2.putText(frame, f"Control Sequence Completed", (7, 35), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (250, 150), (250, 350), (0, 0, 255), 2)
                    cv2.line(frame, (150, 250), (350, 250), (0, 0, 255), 2)

    if len(img) > 0:
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(10)


# function to display camera feed only
def display_camera():
    # initiates and read camera
    img, resx, resy = sim.getVisionSensorCharImage(cam)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resy, resx, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    frame = imutils.resize(img, width=frame_width)
    if len(img) > 0:
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(10)


# ---------- CoppeliaSim initiliasation ----------

# start connection to coppelia
client = RemoteAPIClient()
sim = client.getObject('sim')

# acquire objects
cam = sim.getObject('/Cam')
target = sim.getObject('/target')
floor = sim.getObject('/Floor')

# allow client stepping and begin simulation
client.setStepping(True)
sim.startSimulation()

# wait for 1 second before moving target
while (t := sim.getSimulationTime()) < 1:
    display_camera()
    client.step()  # triggers next simulation step

# original position [0.0, -1.525, 0.99]
target_pos = sim.getObjectPosition(target, -1)
y_pos = target_pos[1]

# gets camera orientation to lock it in place (gimbal)
cam_orient = sim.getObjectOrientation(cam, -1)

# moves drone to pickup location
while target_pos[1] < 0.475:
    display_camera()
    shift_size = 0.05
    y_pos = y_pos + shift_size
    target_pos[1] = y_pos
    sim.setObjectPosition(target, floor, target_pos)
    sim.setObjectOrientation(cam, floor, cam_orient)
    client.step()

# let drone settle for 3 seconds before initiating control
time_current = sim.getSimulationTime()
time_control = time_current + 3
while (t := sim.getSimulationTime()) < time_control:
    detect_markers(0)
    sim.setObjectOrientation(cam, floor, cam_orient)
    client.step()

# control sequence
pos = sim.getObjectPosition(target, -1)
rot = sim.getObjectOrientation(target, -1)
pos[2] = 1
move = 0.0003
rotate = 0.003
descend = 0.0005
dumb = 0
ref_222_y = 0
ref_444_y = 0
length = 0
ori_done_flag = False
x_done_flag = False
y_done_flag = False
alignment_done = False
everything_done = False
try:
    while True:
        detect_markers(1)
        client.step()

except KeyboardInterrupt:
    sim.stopSimulation()
    pass

cv2.destroyAllWindows()
print('Program ended')
