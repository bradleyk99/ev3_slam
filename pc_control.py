#!/usr/bin/env python3
# Program to control the robot within an environment and perform ekf slam

from pc_slam import EV3Connection, EKFSlam, Scans
import math

ev3 = EV3Connection("/dev/rfcomm0")
scans = Scans()
ekf = EKFSlam()

try:
    for step in range(4):
        pose = ekf.get_pose()

        result = ev3.scan(step=5)
        if result is None:
            print("Error during scan")
            continue

        points = scans.scan_to_cartesian(result['scan'], pose)
        corners = scans.detect_corners(points)
        print(f"Detected {len(corners)} corners")

        observations = ekf.corners_to_observations(corners,pose)
        ekf.update(observations)

        # Print current state
        pose = ekf.get_pose()
        print(f"Pose: x={pose[0]:.3f}, y={pose[1]:.3f}, θ={math.degrees(pose[2]):.1f}°")
        print(f"Landmarks: {ekf.n_landmarks}")

        # Rotate 90 degrees
        result = ev3.rotate(speed=10, angle_deg=90)
        if result is None:
            print("Error during rotation")
            continue

        # Compute odometry and predict
        dx, dy, dtheta = scans.odometry(
            result['odometry']['delta_left'],
            result['odometry']['delta_right'],
            ekf.get_pose()[2]
        )
        ekf.predict(dx, dy, dtheta)
        print(f"After rotation: θ={math.degrees(ekf.get_pose()[2]):.1f}°\n")           

    # Final scan
    print("--- Final scan ---")
    pose = ekf.get_pose()
    result = ev3.scan(step=5)
    points = scans.scan_to_cartesian(result['scan'], pose)
    corners = scans.detect_corners(points)
    observations = ekf.corners_to_observations(corners, pose)
    ekf.update(observations)
    
    # Print final state
    print(f"\nFinal pose: x={pose[0]:.3f}, y={pose[1]:.3f}, θ={math.degrees(pose[2]):.1f}°")
    print(f"Total landmarks: {ekf.n_landmarks}")
    print(f"Landmarks: {ekf.get_landmarks()}")
    
    # Plot final state
    scans.plot_map(ekf)
    
    ev3.stop()

except KeyboardInterrupt:
    print("\nInterrupted")
    ev3.stop()

finally:
    ev3.close()
    print("Done")