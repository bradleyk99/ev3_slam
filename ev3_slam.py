#!/usr/bin/env python3
import bluetooth
import json
import time
from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor.lego import UltrasonicSensor
import math

# Adjust ports to match your setup
left_motor = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)
scan_motor = MediumMotor(OUTPUT_A)
ultrasonic = UltrasonicSensor()

WHEEL_RADIUS = 0.0275
WHEEL_BASE = 0.107
TICKS_PER_REV = 360

def send_json(sock, data):
    msg = json.dumps(data) + "\n"
    sock.send(msg.encode("utf-8"))

def recv_json(sock):
    buffer = b""
    while True:
        byte = sock.recv(1)
        if not byte or byte == b"\n":
            break
        buffer += byte
    if buffer:
        return json.loads(buffer.decode("utf-8"))
    return None

def do_scan(start_angle=-90, end_angle=90, step=10):
    """Sweep ultrasonic sensor and return list of [angle, distance_cm]"""
    readings = []
    scan_motor.on_to_position(speed=10, position=start_angle, block=True)
    time.sleep(0.2)
    
    for angle in range(start_angle, end_angle + 1, step):
        scan_motor.on_to_position(speed=10, position=angle, block=True)
        time.sleep(0.5)  # let sensor settle
        dist = ultrasonic.distance_centimeters
        readings.append([angle, dist])
    
    # return to center
    scan_motor.on_to_position(speed=20, position=0, block=True)
    return readings

def do_move(left_speed, right_speed, distance_cm):
    """Move a specified distance using encoder feedback"""
    distance_m = distance_cm / 100.0
    ticks_needed = (distance_m / (2 * math.pi * WHEEL_RADIUS)) * TICKS_PER_REV
    
    left_start = left_motor.position
    right_start = right_motor.position
    
    left_motor.on(speed=left_speed)
    right_motor.on(speed=right_speed)
    
    while True:
        left_delta = abs(left_motor.position - left_start)
        right_delta = abs(right_motor.position - right_start)
        avg_ticks = (left_delta + right_delta) / 2
        if avg_ticks >= ticks_needed:
            break
        time.sleep(0.01)
    
    left_motor.off(brake=True)
    right_motor.off(brake=True)
    
    left_delta = left_motor.position - left_start
    right_delta = right_motor.position - right_start
    return left_delta, right_delta

def do_rotate(speed, angle_deg):
    """Rotate in place using encoder feedback"""
    angle_rad = math.radians(abs(angle_deg))
    arc_length = angle_rad * (WHEEL_BASE / 2)
    ticks_needed = (arc_length / (2 * math.pi * WHEEL_RADIUS)) * TICKS_PER_REV
    
    left_start = left_motor.position
    right_start = right_motor.position
    
    if angle_deg > 0:  # turn right
        left_motor.on(speed=speed)
        right_motor.on(speed=-speed)
    else:  # turn left
        left_motor.on(speed=-speed)
        right_motor.on(speed=speed)
    
    # Wait until we've rotated enough
    while True:
        left_delta = abs(left_motor.position - left_start)
        right_delta = abs(right_motor.position - right_start)
        avg_ticks = (left_delta + right_delta) / 2
        if avg_ticks >= ticks_needed:
            break
        time.sleep(0.01)
    
    left_motor.off(brake=True)
    right_motor.off(brake=True)
    
    left_delta = left_motor.position - left_start
    right_delta = right_motor.position - right_start
    return left_delta, right_delta

def main():
    server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_socket.bind(("", 1))
    server_socket.listen(1)
    print("Waiting for PC connection...")
    
    client_socket, client_info = server_socket.accept()
    print("Connected:", client_info)
    
    # Reset scan motor position as zero
    scan_motor.position = 0
    
    try:
        while True:
            cmd = recv_json(client_socket)
            if cmd is None:
                break
            
            print("Received:", cmd)
            
            if cmd["type"] == "move":
                left_delta, right_delta = do_move(
                    cmd["left_speed"],
                    cmd["right_speed"],
                    cmd["distance_cm"]
                )
                send_json(client_socket, {
                    "type": "move_complete",
                    "odometry": {
                        "delta_left": left_delta,
                        "delta_right": right_delta
                    }
                })
            
            elif cmd["type"] == "rotate":
                left_delta, right_delta = do_rotate(
                    cmd["speed"],
                    cmd["angle_deg"]
                )
                send_json(client_socket, {
                    "type": "rotate_complete",
                    "odometry": {
                        "delta_left": left_delta,
                        "delta_right": right_delta
                    }
                })
            
            elif cmd["type"] == "scan":
                scan_data = do_scan(
                    cmd.get("start_angle", -90),
                    cmd.get("end_angle", 90),
                    cmd.get("step", 10)
                )
                send_json(client_socket, {
                    "type": "scan_complete",
                    "scan": scan_data
                })
            
            elif cmd["type"] == "move_and_scan":
                left_delta, right_delta = do_move(
                    cmd["left_speed"],
                    cmd["right_speed"],
                    cmd["duration_ms"]
                )
                scan_data = do_scan(
                    cmd.get("start_angle", -90),
                    cmd.get("end_angle", 90),
                    cmd.get("step", 10)
                )
                send_json(client_socket, {
                    "type": "move_scan_complete",
                    "odometry": {
                        "delta_left": left_delta,
                        "delta_right": right_delta
                    },
                    "scan": scan_data
                })
            
            elif cmd["type"] == "stop":
                print("Stop command received")
                break
    
    finally:
        left_motor.off()
        right_motor.off()
        client_socket.close()
        server_socket.close()
        print("Connection closed")

if __name__ == "__main__":
    main()