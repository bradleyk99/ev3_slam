from pc_slam import EV3Connection, EKFSlam, Scans, OccupancyGrid, LivePlotter, Explorer, PathPlanner
import math
import time

# Initialize everything
ev3 = EV3Connection("/dev/rfcomm0")
scans = Scans()
ekf = EKFSlam()
occ_grid = OccupancyGrid(size=2.0, resolution=0.1)
trajectory = []
plotter = LivePlotter(occ_grid)
explorer = Explorer(ev3, scans, ekf)
path_planner = PathPlanner(occ_grid, robot_radius=0.01)
explorer.entry_pose = ekf.get_pose().copy()

def scan(steps=5):
    pose = ekf.get_pose()
    trajectory.append(pose.copy())
    
    result = ev3.scan(step=steps)
    if result is None:
        print("Error during scan")
        return None
    
    occ_grid.store_scan(result['scan'], pose)
    points, max_range_angles = scans.scan_to_cartesian(result['scan'], pose)
    occ_grid.update(pose, points, max_range_angles=max_range_angles)
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
    """Execute action returned by wall_follow_step"""
    if action[0] == 'rotate':
        rotate(action[1])
    elif action[0] == 'move':
        move(action[1])
    elif action[0] == 'rotate_move':
        rotate(action[1])
        move(action[2])

def navigate_to_waypoint(target_x, target_y, max_steps=20):
    """Navigate to a waypoint with obstacle avoidance"""
    print(f"  Navigating to ({target_x:.2f}, {target_y:.2f})")
    
    for step in range(max_steps):
        pose = ekf.get_pose()
        rx, ry, rtheta = pose
        
        # Distance and angle to target
        dx = target_x - rx
        dy = target_y - ry
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.10:
            print(f"    Reached waypoint!")
            return True
        
        # Scan for obstacles
        result = ev3.scan(step=10)
        if result is None:
            continue
        
        # Update EKF and map
        points, max_range_angles = scans.scan_to_cartesian(result['scan'], pose)
        occ_grid.store_scan(result['scan'], pose)
        occ_grid.update(pose, points, max_range_angles=max_range_angles, max_range=1.0)
        trajectory.append(pose.copy())
        
        corners = scans.detect_corners(points)
        observations = ekf.corners_to_observations(corners, pose)
        ekf.update(observations)
        plotter.update(ekf, trajectory)
        
        # Get distances
        front_dist, right_dist, left_dist = explorer.get_distances(result, pose)
        print(f"    Step {step+1}: dist={dist:.2f}m, F={front_dist:.2f}m, R={right_dist:.2f}m, L={left_dist:.2f}m")
        
        # Obstacle check first
        if front_dist < 0.15:
            print("    Obstacle ahead! Avoiding...")
            if left_dist > right_dist:
                rotate(-45)
            else:
                rotate(45)
            continue
        
        # Angle to target
        target_angle = math.atan2(dy, dx)
        angle_diff = explorer.normalize_angle(target_angle - rtheta)
        
        if abs(angle_diff) > math.radians(20):
            turn = -math.degrees(angle_diff)  # Negate for EV3 convention
            turn = max(-60, min(60, turn))
            print(f"    Turning {turn:.1f}°")
            rotate(turn)
        else:
            move_dist = min(15, dist * 100)
            move_dist = max(5, move_dist)
            print(f"    Moving {move_dist:.0f}cm")
            move(move_dist)
    
    print(f"    Failed to reach waypoint after {max_steps} steps")
    return False

def find_exit_in_map():
    """Rebuild map and search for exits"""
    pose = ekf.get_pose()
    
    # Rebuild map with corrected poses
    #print("  Rebuilding map...")
    #occ_grid.rebuild_with_ekf(trajectory)
    
    # Find gaps
    gaps = path_planner.find_gaps_in_grid(pose)
    print(f"  Found {len(gaps)} potential gaps")
    
    # Filter out gaps near entry
    valid_gaps = []
    for gap in gaps:
        dist_to_entry = math.sqrt(
            (gap['center'][0] - explorer.entry_pose[0])**2 +
            (gap['center'][1] - explorer.entry_pose[1])**2
        )
        if dist_to_entry > 0.2:
            valid_gaps.append(gap)
            print(f"    Gap at ({gap['center'][0]:.2f}, {gap['center'][1]:.2f}), dist={gap['distance']:.2f}m")
        else:
            print(f"    Ignoring gap near entry")
    
    return valid_gaps

def navigate_to_exit(goal):
    """Use A* to plan and execute path to exit"""
    pose = ekf.get_pose()
    goal_x, goal_y = goal['center']
    
    print(f"\nPlanning path from ({pose[0]:.2f}, {pose[1]:.2f}) to ({goal_x:.2f}, {goal_y:.2f})")
    
    # Plan path
    path = path_planner.astar(pose, (goal_x, goal_y))
    
    if path is None:
        print("No path found!")
        path_planner.plot_inflated(pose,None,(goal_x,goal_y))
        return False

    # Simplify path
    waypoints = path_planner.simplify_path(path)
    print(f"Path has {len(waypoints)} waypoints")
    
    # Visualize path
    path_planner.plot_inflated(pose, waypoints, (goal_x, goal_y), filename='planned_path.png')
    
    # Navigate through waypoints
    for i, (wx, wy) in enumerate(waypoints):
        print(f"\nWaypoint {i+1}/{len(waypoints)}: ({wx:.2f}, {wy:.2f})")
        
        success = navigate_to_waypoint(wx, wy)
        
        if not success:
            print("  Replanning...")
            
            # Rebuild map and replan
            # occ_grid.rebuild_with_ekf(trajectory)
            pose = ekf.get_pose()
            new_path = path_planner.astar(pose, (goal_x, goal_y))
            
            if new_path:
                waypoints = path_planner.simplify_path(new_path)
                print(f"  New path has {len(waypoints)} waypoints")
            else:
                print("  Replanning failed!")
                return False
    
    # Move forward to clear the exit
    print("\nClearing exit...")
    move(30)
    
    return True

# ============= MAIN =============
try:
    exit_found = None
    max_steps = 50
    check_interval = 1  # Check for exits every N steps
    
    # Scan then move forward
    #print("Preparing to enter arena")
    #move(30)

    # Start with 360 degree scan
    #print("Scanning surroundings...")
    """
    for step in range(4):
        scan()
        rotate(angle=90)
    """

    while (True):
        print("=== Phase 1: Exploration ===\n")
        for step in range(max_steps):
            print(f"Step {step + 1}")
            
            result = scan()
            if result is None:
                continue
            
            pose = ekf.get_pose()
            
            # Periodically check map for exit
            
            if step > 6 and step % check_interval == 0:
                print("\n  --- Checking map for exits ---")
                gaps = find_exit_in_map()
    
                if gaps:
                    # Use closest gap
                    exit_found = gaps[0]
                    print(f"  Exit selected at ({exit_found['center'][0]:.2f}, {exit_found['center'][1]:.2f})")
                    break
                else:
                    print("  No valid exits found yet, continuing exploration...")
            
            # Wall follow to explore
            action = explorer.wall_follow_step(result, pose, wall_dist=0.2, side='right',dist=15, Kp=200)
            execute_action(action)
            
            # Check if completed full loop
            if step > 25:
                dist_to_entry = math.sqrt(
                    (pose[0] - explorer.entry_pose[0])**2 +
                    (pose[1] - explorer.entry_pose[1])**2
                )
                if dist_to_entry < 0.25:
                    print("\nCompleted loop around arena")
                    
                    # Final check for exits
                    gaps = find_exit_in_map()
                    if gaps:
                        exit_found = gaps[0]
                    break
        
        print(f"\n=== Exploration Complete ===")
        print(f"Steps: {step + 1}")
        print(f"Landmarks: {ekf.n_landmarks}")
        print(f"Scans stored: {len(occ_grid.scan_history)}")
        
        # Phase 2: Navigate to exit
        if exit_found:
            print("\n=== Phase 2: Navigate to Exit ===")
            
            # Visualize gaps
            pose = ekf.get_pose()
            all_gaps = path_planner.find_gaps_in_grid(pose)
            if all_gaps:
                path_planner.plot_gaps(pose, all_gaps)
            
            # Navigate to exit
            success = navigate_to_exit(exit_found)
            
            if success:
                print("\n*** ESCAPED! ***")
                break
            else:
                print("\nFailed to reach exit, returning to exploring")
                continue
        else:
            print("\nNo exit found during exploration")
            break

    # Final map
    print("\nGenerating final map...")
    # occ_grid.rebuild_with_ekf(trajectory)
    occ_grid.plot(ekf, trajectory)
    
    input("\nPress Enter to close...")
    ev3.stop()

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    
    if len(trajectory) > 0:
        print("Generating partial map...")
        occ_grid.plot(ekf, trajectory)
    
    ev3.stop()

finally:
    ev3.close()
    print("Done")