from __future__ import print_function
import cv2
import sys
import time
import imutils
import argparse
import keyboard
import numpy as np
from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobalRelative

# RPi-only modules:
# import RPi.GPIO as GPIO
# import adafruit_vl6180x
# import busio
# import board

print('Program started')

# ---------- ArUco Initiliasation ----------

# include camera matrix and distortion coefficients from camera calibration + marker sizes
matrix_coefficients = np.array([[966.90841671, 0, 611.58804072],
                                [0, 965.65159698, 371.44866427],
                                [0, 0, 1]])
distortion_coefficients = np.array([[6.86566894e-2, -1.79345595, 2.17383609e-3, -1.84014871e-3, 1.17946050e+1]])

marker_size = 8  # centimeters
frame_width = 1000  # for opencv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args1 = vars(ap.parse_args())

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
if ARUCO_DICT.get(args1["type"], None) is None:
    print("ArUCo tag of '{}' is not supported".format(args1["type"]))
    sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("Detecting '{}' tags".format(args1["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args1["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# ---------- DroneKit Initialisation ----------

# Set up option parsing to get connection string
parser = argparse.ArgumentParser(description='Commands vehicle using vehicle.simple_goto.')
parser.add_argument('--connect',
                    help="Vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True, baud=57600)
vehicle.wait_ready(True, raise_exception=False)


# Flight Commands Definitions
def arm_and_takeoff(target_alt):
    # Arms and fly drone to target altitude
    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(3)

    print("Taking off!")
    vehicle.simple_takeoff(target_alt)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto
    #  (otherwise the command after Vehicle.simple_takeoff will execute immediately)
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        # Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt >= target_alt * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)


def set_velocity_body(v, vx, vy, vz):
    # Note that vz is positive downward
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,  # -- BITMASK -> Consider only the velocities
        0, 0, 0,  # -- POSITION
        vx, vy, vz,  # -- VELOCITY
        0, 0, 0,  # -- ACCELERATIONS
        0, 0)
    v.send_mavlink(msg)


def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    # Move vehicle in direction based on specified velocity vectors
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 50 Hz cycle
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(0.02)


def land():
    print("Vehicle in LAND mode")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.location.global_relative_frame.alt > 0:
        if vehicle.location.global_relative_frame.alt > 0.1:
            print("Altitude: ", vehicle.location.global_relative_frame.alt)
            set_velocity_body(vehicle, 0, 0, 0.1)
        else:
            break
        time.sleep(1)


def condition_yaw(degree, spin=1, relative=False):
    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,  # confirmation
        degree,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        spin,          # param 3, direction -1 ccw, 1 cw
        is_relative,  # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)


def wait(t):
    for i in np.arange(0, t):
        if vehicle.mode.name == "STABILIZE":
            sys.exit()
        else:
            time.sleep(1)


# ---------- CV Initiliasation ----------

# initialize the video stream and allow the camera sensor to warm up
print("Starting video stream")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# initiliase centres and fps
cx222, cy222, cx444, cy444 = 0, 0, 0, 0

prev_frame_time = 0
new_frame_time = 0


def detect_markers():
    global cx444, cy444, ref_222_y, ref_444_y, ref_444_x, ref_222_x, avgx, avgy, rotation, pause, prev_frame_time,\
        ori_start_flag, ori_done_flag, x_done_flag, y_done_flag,\
        everything_done, control_done, alignment_done, activate_control, spin
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=frame_width)
        if not ret:
            break

        new_frame_time = time.time()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect ArUco markers in the input frame
        marker_corners, marker_ids, reject = cv2.aruco.detectMarkers(gray_frame,
                                                                     arucoDict,
                                                                     parameters=arucoParams)
        # loop over the detected ArUCo corners
        if not control_done:
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
                    distance = np.sqrt(tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2 + tvec[i][0][2] ** 2)

                    # Draw the pose of the marker
                    if ids == 444:
                        cx444 = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cy444 = int((topLeft[1] + bottomRight[1]) / 2.0)
                        avgx = round(tvec[i][0][0], 1)
                        avgy = round(tvec[i][0][1], 1)
                        cv2.putText(frame, f"id: {ids} Dist: {round(distance, 2)}", top_right,
                                    cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"x:{avgx} y: {avgy} ", bottom_right,
                                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                    elif ids == 222:
                        cv2.putText(frame, f"id: {ids}", top_right,
                                    cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)

                        # compute and draw the center (x, y)-coordinates of both ArUco markers
                        cx222 = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cy222 = int((topLeft[1] + bottomRight[1]) / 2.0)
                        cx = int((cx222 + cx444) / 2)
                        cy = int((cy222 + cy444) / 2)
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    if not activate_control:
                        pass
                    else:
                        if not everything_done:
                            # Orientation Process
                            reference1 = 444  # Reference tag for orientation
                            reference2 = 222  # Reference tag for orientation

                            cv2.putText(frame, f"Control Sequence Initiated:", (7, 35), cv2.FONT_HERSHEY_PLAIN,
                                        2, (0, 255, 0), 2, cv2.LINE_AA)

                            # draw crosshair
                            cv2.line(frame, (500, 225), (500, 525), (0, 0, 255), 2)
                            cv2.line(frame, (350, 375), (650, 375), (0, 0, 255), 2)

                            if not ori_start_flag:  # coord acquisition is not very good
                                for j in range(marker_ids.size):
                                    if marker_ids[j][0] == reference1:
                                        ref_444_x = tvec[j][0][0]
                                        ref_444_y = -tvec[j][0][1]
                                    elif marker_ids[j][0] == reference2:
                                        ref_222_x = tvec[j][0][0]
                                        ref_222_y = -tvec[j][0][1]

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
                                print("Rotate: %s" % rotation)
                                pause = time.time() + 5
                                ori_start_flag = True

                            if not ori_done_flag:
                                if -5 < rotation < 5:
                                    print('Rotation Complete')
                                    ori_done_flag = True
                                else:
                                    cv2.putText(frame, f"Drone Rotating", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                2, (0, 255, 0), 2, cv2.LINE_AA)
                                    if rotation > 0:
                                        spin = 1
                                    if rotation < 0:
                                        spin = -1
                                    condition_yaw(abs(rotation), spin=spin, relative=True)
                                    if time.time() < pause:
                                        pass
                                    else:
                                        ori_done_flag = True

                            if ori_done_flag:
                                if x_done_flag == False and alignment_done == False:
                                    if avgx > 0.5:
                                        cv2.putText(frame, f"Drone Moving Right", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 255, 0), 2, cv2.LINE_AA)
                                        send_ned_velocity(0, 0.5, 0, 0.02)
                                    elif avgx < -0.5:
                                        cv2.putText(frame, f"Drone Moving Left", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                     2, (0, 255, 0), 2, cv2.LINE_AA)
                                        send_ned_velocity(0, -0.5, 0, 0.02)
                                    else:
                                        x_done_flag = True

                                # Alignement in the y-axis
                                if x_done_flag == True and alignment_done == False:
                                    if avgx > 0.5 or avgx < -0.5:
                                        x_done_flag = False
                                        alignment_done = False
                                    else:
                                        pass
                                    if avgy > 0.5:
                                        cv2.putText(frame, f"Drone Moving Forward", (7, 70), cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 255, 0), 2, cv2.LINE_AA)
                                        send_ned_velocity(0.5, 0, 0, 0.02)
                                    elif avgy < -0.5:
                                        cv2.putText(frame, f"Drone Moving Backward", (7, 70),
                                                    cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 255, 0), 2, cv2.LINE_AA)
                                        send_ned_velocity(-0.5, 0, 0, 0.02)
                                    else:
                                        y_done_flag = True
                                        alignment_done = True

                            if y_done_flag:
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
                                    send_ned_velocity(0, 0, 0.5, 0.02)
                                else:
                                    everything_done = True
                            else:
                                pass
                        else:
                            control_done = True

        else:
            cv2.putText(frame, f"Control Sequence Completed", (7, 35), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2, cv2.LINE_AA)

        # Camera fps
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, f"FPS:{fps}", (820, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print('Control manually terminated')
            break

        elif key == ord("g"):
            print('Control algorithm activated')
            activate_control = True

    # Do a bit of cleanup
    cv2.destroyAllWindows()


# ---------- Interface Initialisation ----------
"""
# Servo pin setup
servo_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(2.5)

# Setup I2C board
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_vl6180x.VL6180X(i2c)


def connect_interface():
    range = sensor.range

    # If the range is less than or equal to 20, set the servo position to 135
    if range <= 20:
        pwm.ChangeDutyCycle(10.0)
        time.sleep(0.5)
        pwm.stop()
        GPIO.cleanup()
    else:
        print("[Error] Interface connection error")
        sys.exit()


def disconnect_interface():
    pwm.ChangeDutyCycle(12.5)
    time.sleep(0.1)
    pwm.stop()
    GPIO.cleanup()
"""

# ---------- Flags Initialisation ----------

activate_control = False
ori_start_flag = False
ori_done_flag = False
x_done_flag = False
y_done_flag = False
alignment_done = False
everything_done = False
control_done = False
connect_done = False

# ---------- Main Program ----------

arm_and_takeoff(2)

print("Set default target airspeed to 0.5m/s")
vehicle.airspeed = 0.5

print("Forward 5 seconds at 0.1m/s")
send_ned_velocity(0.5, 0, 0, 5)

print("Pause")
wait(3)

# Control sequence, 'g' to start, 'q' to terminate
detect_markers()

# print("Attempt interface connection")
# connect_interface()

# print("Climb 5 seconds at 0.4m/s")
# send_ned_velocity(0, 0, -0.4, 5)

# command drone to move to dropoff location and descend

# while True:
#     # drone loiters until key press
#     if keyboard.is_pressed("z"):
#         print("Dropping package")
#         # disconnect_interface()
#         break

# command drone to RTH

land()
print("Land Success")
time.sleep(5)
vehicle.armed = False
time.sleep(5)
vehicle.close()
print("Program ended")
sys.exit()
