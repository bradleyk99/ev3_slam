from pc_slam import EV3Connection, EKFSlam, Scans, OccupancyGrid, LivePlotter, Explorer
import math
import time

ev3 = EV3Connection("/dev/rfcomm0")
scans = Scans()
ekf = EKFSlam()
occ_grid = OccupancyGrid(size=2.0, resolution=0.05)
trajectory = []
plotter = LivePlotter(occ_grid)
explorer = Explorer(ev3, scans, ekf)
explorer.entry_pose = ekf.get_pose().copy()

def scan(steps=5):
    pose = ekf.get_pose()
    trajectory.append(pose.copy())
    
    result = ev3.scan(step=steps)
    if result is None:
        print("Error during scan")
        return False
    
    # Store raw scan for later rebuild
    occ_grid.store_scan(result['scan'], pose)

    points = scans.scan_to_cartesian(result['scan'], pose)
    occ_grid.update(pose, points)
    plotter.update(ekf, trajectory)
    
    corners = scans.detect_corners(points)
    print(f"Detected {len(corners)} corners")
    
    observations = ekf.corners_to_observations(corners, pose)
    ekf.update(observations)
    plotter.update(ekf, trajectory)
    return result

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

def execute_action(action):
    # Excecute action returned by wall follow step
    if action[0] == 'rotate':
        rotate(action[1])
    elif action[0] == 'move':
        move(action[1])
    elif action[0] == 'rotate_move':
        rotate(action[1])
        move(action[2])

try:
    exit_found = None
    max_steps = 50

    for step in range(max_steps):
        print(f"Step {step +1}")
        result = scan()
        if result is None:
            continue
        
        pose = ekf.get_pose()

        # Check for exit after first few steps
        if step > 3:
            gaps = explorer.detect_gaps(result, pose, min_gap_width=0.3)

            for gap in gaps:
                near_entry = explorer.is_near_entry(pose)
                print(f"  Gap check: near_entry={near_entry}, width={gap['width']:.2f}, dist={gap['distance']:.2f}")
                
                if not near_entry or near_entry:
                    if gap['width'] > 0.3 and gap['distance'] < 1:
                        print("*** EXIT FOUND! ***")
                        exit_found = gap
                        break
                else:
                    print(f"  Skipping - too close to entry")

        if exit_found:
            break

        # Wall follow
        action = explorer.wall_follow_step(result,pose,wall_dist=0.2,side='right',Kp=200)
        execute_action(action)

        # Check if completed loop
        if step > 20 and explorer.is_near_entry(pose, threshold=0.2):
            print("Completed loop, no exit found")
            break
    
    print("Exploration Complete")
    print(f"Steps: {step + 1}")
    print(f"Landmarks: {ekf.n_landmarks}")

    if exit_found:
        print("Navigating to exit")
        explorer.go_through_gap(exit_found)
        print("Complete")
    
    print("Rebuilding map")
    occ_grid.rebuild_with_ekf(trajectory)
    occ_grid.plot(ekf,trajectory)

    input("Press Enter to End")
            
    ev3.stop()

except KeyboardInterrupt:
    print("\nInterrupted")
    ev3.stop()

finally:
    plotter.close()
    ev3.close()
    print("Done")