from pc_slam import EV3Connection, EKFSlam, Scans, OccupancyGrid, LivePlotter
import math
import time

ev3 = EV3Connection("/dev/rfcomm0")
scans = Scans()
ekf = EKFSlam()
occ_grid = OccupancyGrid(size=2.0, resolution=0.05)
trajectory = []
plotter = LivePlotter(occ_grid)

def scan(steps=5):
    pose = ekf.get_pose()
    trajectory.append(pose.copy())
    
    result = ev3.scan(step=steps)
    if result is None:
        print("Error during scan")
        return False
    
    points = scans.scan_to_cartesian(result['scan'], pose)
    occ_grid.update(pose, points)
    plotter.update(ekf, trajectory)
    
    corners = scans.detect_corners(points)
    print(f"Detected {len(corners)} corners")
    
    observations = ekf.corners_to_observations(corners, pose)
    ekf.update(observations)
    plotter.update(ekf, trajectory)
    return True

def move(distance, speed=10):
    result = ev3.move(left_speed=speed, right_speed=speed, distance_cm=distance)
    if result is None:
        print("Error during move")
        return False
    
    dx, dy, dtheta = scans.odometry(
        result['odometry']['delta_left'],
        result['odometry']['delta_right'],
        ekf.get_pose()[2]
    )
    ekf.predict(dx, dy, dtheta)
    
    pose = ekf.get_pose()
    trajectory.append(pose.copy())
    plotter.update(ekf, trajectory)
    return True

def rotate(angle, speed=10):
    result = ev3.rotate(speed=speed, angle_deg=angle)
    if result is None:
        print("Error during rotation")
        return False
    
    dx, dy, dtheta = scans.odometry(
        result['odometry']['delta_left'],
        result['odometry']['delta_right'],
        ekf.get_pose()[2]
    )
    ekf.predict(dx, dy, dtheta)
    plotter.update(ekf, trajectory)
    return True

try:
    for step in range(4):
        print(f"=== Step {step + 1} ===")
        scan()
        move(20)
        scan()
        move(20)
        scan()
        rotate(90)
    
    print("--- Done ---")
    print(f"Final landmarks: {ekf.n_landmarks}")
    input("Press Enter to close...")
    ev3.stop()

except KeyboardInterrupt:
    print("\nInterrupted")
    ev3.stop()

finally:
    plotter.close()
    ev3.close()
    print("Done")