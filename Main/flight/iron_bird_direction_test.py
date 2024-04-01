from __future__ import print_function
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

# Set up option parsing to get connection string
import argparse

parser = argparse.ArgumentParser(description='Commands vehicle using vehicle.simple_goto.')
parser.add_argument('--connect',
                    help="Vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect
sitl = None

manual_arm = True

# Start SITL if no connection string specified
if not connection_string:
    from dronekit_sitl import SITL

    sitl = SITL()
    sitl.download('copter', '3.3', verbose=True)
    sitl_args = ['-I0', '--model', 'quad', '--home=50.937257,-1.405070,0,45']
    sitl.launch(sitl_args, await_ready=True, restart=True)
    connection_string = sitl.connection_string()

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)


def arm_and_takeoff(target_alt):
    # Arms and fly drone to target altitude
    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arm motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")

    while vehicle.mode != 'GUIDED':
        print("Waiting for drone to enter GUIDED flight mode")
        time.sleep(3)
        print("Vehicle now in GUIDED MODE, arm to begin")

        if not manual_arm:
            vehicle.armed = True
            while not vehicle.armed:
                print("Automatically arming...")
                time.sleep(3)
        else:
            if not vehicle.armed:
                print("Waiting for RC arming...")
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


def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 50 Hz cycle
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(0.02)


def condition_yaw(degree, spin=1, relative=False):
    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,  # confirmation
        degree,  # param 1, yaw in degrees
        0,  # param 2, yaw speed deg/s
        spin,  # param 3, direction -1 ccw, 1 cw
        is_relative,  # param 4, relative offset 1, absolute angle 0
        0, 0, 0)  # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)


# Main program starts
# arm_and_takeoff(5)
# print("Hovering for 3 seconds...")
# time.sleep(3)
# print("Testing Directoinal Commands:")
# time.sleep(1)
# print("Forward 3 seconds")
# send_ned_velocity(1, 0, 0, 3)
# print("Pause")
# time.sleep(5)
# print("Backward 3 seconds")
# send_ned_velocity(-1, 0, 0, 3)
# print("Pause")
# time.sleep(5)
# print("Right 3 seconds")
# send_ned_velocity(0, 1, 0, 3)
# print("Pause")
# time.sleep(5)
# print("Left 3 seconds")
# send_ned_velocity(0, -1, 0, 3)
# print("Pause")
# time.sleep(5)
# print("Rotate Right 90 Degrees")
# condition_yaw(90, spin=1, relative=True)  # spin=1 for right
# print("Pause")
# time.sleep(5)
# print("Rotate Left 90 Degrees")
# condition_yaw(90, spin=-1, relative=True)
# print("Pause")
# time.sleep(5)

arm_and_takeoff(1)
time.sleep(10)

# Land and disarm
land()
print("Land Success")
time.sleep(5)
vehicle.armed = False
time.sleep(5)
vehicle.close()
exit()
