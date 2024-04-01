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

# Start SITL if no connection string specified
if not connection_string:
    from dronekit_sitl import SITL
    sitl = SITL()
    sitl.download('copter', '3.3', verbose=True)
    sitl_args = ['-I0', '--model', 'quad', '--home=50.937257,-1.405070,0,180']
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
        0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_NED,
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


# Main program starts
arm_and_takeoff(5)

print("Set default/target airspeed to 5")
vehicle.airspeed = 5

# Move to 50.937136,-1.405223 (begins.sector.truck)
print("Going towards pickup")
point1 = LocationGlobalRelative(50.937136, -1.405223, 5)
vehicle.simple_goto(point1)
while True:
    print(vehicle.location.global_frame)
    # Break and return from function just after reaching pickup.
    if vehicle.location.global_frame.lon <= -1.405225 and\
            vehicle.location.global_relative_frame.lat <= 50.937138:
        print("Reached pickup location")
        break
    time.sleep(1)

print("Hovering for 5 seconds...")
time.sleep(5)

# Land and disarm
land()
print("Land Success")
time.sleep(5)
vehicle.armed = False
time.sleep(5)
vehicle.close()
