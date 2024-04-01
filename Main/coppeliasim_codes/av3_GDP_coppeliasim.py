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
marker_size = 8  # centimeters
frame_width = 500  # for opencv

# camera coord fudge values
xadj = 46
yadj = -46

# initiliase centres
x222, y222, x444, y444 = 0, 0, 0, 0
cx222, cy222, cx444, cy444 = 0, 0, 0, 0

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
    global x_done_flag, alignment_done, threshold_ori, \
        ref_444_y, ref_222_y, y_done_flag, reference1, ori_done_flag, \
        cam_orient_new, xadj, yadj, everything_done, ref_444_x, ref_222_x, rotation, ori_start_flag, time_c, cx444, cy444, cx222, cy222, x444, y444, avgx, avgy
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
            x = tvec[i][0][0] + xadj
            y = -tvec[i][0][1] + yadj
            z = tvec[i][0][2] - 40
            distance = np.sqrt(z ** 2 + x ** 2 + y ** 2)

            # Draw the pose of the marker
            if ids == 444:
                cx444 = int((topLeft[0] + bottomRight[0]) / 2.0)
                cy444 = int((topLeft[1] + bottomRight[1]) / 2.0)
                x444 = round(tvec[i][0][0] + xadj, 1)
                y444 = round(-tvec[i][0][1] + yadj, 1)
                cv2.putText(frame, f"id: {ids} Dist: {round(distance, 2)}", top_right,
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.putText(frame, f"x:{round(tvec[i][0][0] + xadj, 1)} y: {round(-tvec[i][0][1] + yadj, 1)} ",
                #             bottom_right,
                #             cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if ids == 222:
                cv2.putText(frame, f"id: {ids}", top_right,
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)

                # compute and draw the center (x, y)-coordinates of both ArUco markers
                cx222 = int((topLeft[0] + bottomRight[0]) / 2.0)
                cy222 = int((topLeft[1] + bottomRight[1]) / 2.0)
                x222 = round(tvec[i][0][0] + xadj, 1)
                y222 = round(-tvec[i][0][1] + yadj, 1)
                cx = int((cx222 + cx444) / 2)
                cy = int((cy222 + cy444) / 2)
                avgx = round((x222 + x444)/ 2, 1)
                avgy = round((y222 + y444) / 2, 1)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"x:{avgx} y: {avgy}",
                            bottom_right,
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

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

                    if not ori_start_flag:  # coord acquisition is not very good
                        for j in range(marker_ids.size):
                            if marker_ids[j][0] == reference1:
                                ref_444_x = tvec[j][0][0] + xadj
                                ref_444_y = -tvec[j][0][1] + yadj
                                print(ref_444_x, ref_444_y)
                            elif marker_ids[j][0] == reference2:
                                ref_222_x = tvec[j][0][0] + xadj
                                ref_222_y = -tvec[j][0][1] + yadj
                                print(ref_222_x, ref_222_y)

                        if ref_222_y > ref_444_y:
                            if ref_444_x > ref_222_x:
                                rotation = (np.arctan((ref_222_y - ref_444_y) / (ref_444_x - ref_222_x))
                                            * (180 / np.pi))
                            elif ref_444_x < ref_222_x:
                                rotation = 180 + (np.arctan((ref_222_y - ref_444_y) / (ref_444_x - ref_222_x))
                                                  * (180 / np.pi))
                        elif ref_222_y < ref_444_y:
                            if ref_444_x > ref_222_x:
                                rotation = -(np.arctan((ref_444_y - ref_222_y) / (ref_444_x - ref_222_x))
                                             * (180 / np.pi))
                            elif ref_444_x < ref_222_x:
                                rotation = -(180 + (np.arctan((ref_444_y - ref_222_y) / (ref_444_x - ref_222_x))
                                                    * (180 / np.pi)))
                        print(rotation)
                        ori_start_flag = True

                    if not ori_done_flag:
                        if -5 < rotation < 5:
                            print('Rotation Complete')
                            time_c = sim.getSimulationTime() + 1
                            cam_orient_new = sim.getObjectOrientation(cam, -1)
                            ori_done_flag = True
                        elif abs(rot[2] * (180 / np.pi)) < abs(rotation * 0.99):
                            rot[2] = rot[2] - ((rotation / 1e4) * (
                                (abs(rotation) - abs(rot[2] * (180 / np.pi)))) / abs(rotation))
                            cv2.putText(frame, f"Drone Rotating", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                        2, (0, 255, 0), 2, cv2.LINE_AA)
                            sim.setObjectOrientation(target, floor, rot)
                        else:
                            print('Rotation Complete')
                            time_c = sim.getSimulationTime() + 1
                            cam_orient_new = sim.getObjectOrientation(cam, -1)
                            ori_done_flag = True

                    if ori_done_flag:
                        sim.setObjectOrientation(cam, floor, cam_orient_new)
                        time_p = sim.getSimulationTime()
                        if time_p < time_c:
                            cv2.putText(frame, f"Drone Stabilising", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                        2, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            # direction is not relative to body
                            if x_done_flag == False and alignment_done == False:
                                if avgx > 0.5:
                                    cv2.putText(frame, f"Drone Moving Right", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                2, (0, 255, 0), 2, cv2.LINE_AA)
                                    pos[0] = pos[0] + (move * np.cos(rotation * np.pi / 180))
                                    pos[1] = pos[1] - (move * np.sin(rotation * np.pi / 180))
                                    sim.setObjectPosition(target, floor, pos)
                                elif avgx < -0.5:
                                    cv2.putText(frame, f"Drone Moving Left", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                2, (0, 255, 0), 2, cv2.LINE_AA)
                                    pos[0] = pos[0] - (move * np.cos(rotation * np.pi / 180))
                                    pos[1] = pos[1] + (move * np.sin(rotation * np.pi / 180))
                                    sim.setObjectPosition(target, floor, pos)
                                else:
                                    x_done_flag = True

                            # Alignement in the y-axis
                            if x_done_flag == True and alignment_done == False:
                                sim.setObjectOrientation(cam, floor, cam_orient_new)
                                if avgx > 0.5 or avgx < -0.5:
                                    x_done_flag = False
                                    alignment_done = False
                                else:
                                    pass
                                if avgy > 0.5:
                                    cv2.putText(frame, f"Drone Moving Forward", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                2, (0, 255, 0), 2, cv2.LINE_AA)
                                    pos[0] = pos[0] + (move * np.sin(rotation * np.pi / 180))
                                    pos[1] = pos[1] + (move * np.cos(rotation * np.pi / 180))
                                    sim.setObjectPosition(target, floor, pos)
                                elif avgy < -0.5:
                                    cv2.putText(frame, f"Drone Moving Backward", (7, 70),
                                                cv2.FONT_HERSHEY_PLAIN,
                                                2, (0, 255, 0), 2, cv2.LINE_AA)
                                    pos[0] = pos[0] - (move * np.sin(rotation * np.pi / 180))
                                    pos[1] = pos[1] - (move * np.cos(rotation * np.pi / 180))
                                    sim.setObjectPosition(target, floor, pos)
                                else:
                                    y_done_flag = True
                                    alignment_done = True

                        if y_done_flag:
                            sim.setObjectOrientation(cam, floor, cam_orient_new)
                            if avgx > 0.5 or avgx < -0.5:
                                x_done_flag = False
                                y_done_flag = False
                                alignment_done = False
                            elif avgy > 0.5 or avgy < -0.5:
                                y_done_flag = False
                                alignment_done = False
                            else:
                                pass

                            # Height Control Process
                            if distance >= 35:
                                cv2.putText(frame, f"Drone Descending", (7, 100), cv2.FONT_HERSHEY_PLAIN,
                                            2, (0, 255, 0), 2, cv2.LINE_AA)
                                pos[2] = pos[2] - descend
                                sim.setObjectPosition(target, floor, pos)
                                xadj = xadj - 0.0285
                                yadj = yadj + 0.0282
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
descend = 0.0005
dumb = 0
ref_222_x = 0
ref_222_y = 0
ref_444_x = 0
ref_444_y = 0
rotation = 0
length = 0
ori_start_flag = False
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
