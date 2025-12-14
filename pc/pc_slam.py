#!/usr/bin/env python3
import json
import termios
import time
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

class EV3Connection:
    def __init__(self, device="/dev/rfcomm0"):
        print(f"Opening connection to {device}...")
        self.fd = open(device, "r+b", buffering=0)
        attrs = termios.tcgetattr(self.fd.fileno())
        attrs[3] = attrs[3] & ~termios.ECHO
        termios.tcsetattr(self.fd.fileno(), termios.TCSANOW, attrs)
        print("Connected!")
    
    def send(self, data):
        msg = json.dumps(data) + "\n"
        self.fd.write(msg.encode("utf-8"))
        self.fd.flush()
    
    def recv(self):
        buffer = b""
        while True:
            byte = self.fd.read(1)
            if not byte or byte == b"\n":
                break
            buffer += byte
        if buffer:
            return json.loads(buffer.decode("utf-8"))
        return None
    
    def move(self, left_speed, right_speed, distance_cm):
        self.send({
            "type": "move",
            "left_speed": left_speed,
            "right_speed": right_speed,
            "distance_cm": distance_cm
        })
        return self.recv()
    
    def rotate(self, speed, angle_deg):
        self.send({
            "type": "rotate",
            "speed": speed,
            "angle_deg": angle_deg
        })
        return self.recv()
    
    def scan(self, start_angle=-120, end_angle=120, step=10):
        self.send({
            "type": "scan",
            "start_angle": start_angle,
            "end_angle": end_angle,
            "step": step
        })
        return self.recv()
    
    def move_and_scan(self, left_speed, right_speed, distance_cm,
                      start_angle=-120, end_angle=120, step=10):
        self.send({
            "type": "move_and_scan",
            "left_speed": left_speed,
            "right_speed": right_speed,
            "distance_cm": distance_cm,
            "start_angle": start_angle,
            "end_angle": end_angle,
            "step": step
        })
        return self.recv()
    
    def stop(self):
        self.send({"type": "stop"})
    
    def close(self):
        self.fd.close()

class Scans:
    def __init__(self):
        # Robot parameters
        self.WHEEL_RADIUS = 0.0275 # meters
        self.WHEEL_BASE = 0.107 # meters
        self.TICKS_PER_REV = 360

    def ticks_to_meters(self, ticks):
        # Convert encoder ticks to distance traveled
        return (ticks / self.TICKS_PER_REV) * 2 * math.pi * self.WHEEL_RADIUS

    def odometry(self, delta_left, delta_right, theta):   
        # Convert encoder deltas to pose change
        dist_left = self.ticks_to_meters(delta_left)
        dist_right = self.ticks_to_meters(delta_right)
        dist_center = (dist_left + dist_right)/2

        dtheta = (dist_right - dist_left)/self.WHEEL_BASE
        mid_theta = theta + dtheta / 2 # More accurate to reduce drift

        dx = dist_center * math.cos(mid_theta)
        dy = dist_center * math.sin(mid_theta)

        return dx, dy, dtheta

    def update_pose(self, pose, result):
        dx, dy, dtheta = self.odometry(result['odometry']['delta_left'],
                                       result['odometry']['delta_right'],
                                       pose[2])
        pose[0] += dx
        pose[1] += dy
        pose[2] += dtheta

    def scan_to_cartesian(self, scan, pose):
        # Convert scan to coordinates
        rx, ry, rtheta = pose
        points = []

        for angle_deg, distance_cm in scan:
            if distance_cm > 150:   # Ignore readings above 150cm - unreliable
                continue

            # Convert to robot frame
            angle_rad = math.radians(-angle_deg)
            local_x = (distance_cm/100) * math.cos(angle_rad)
            local_y = (distance_cm/100) * math.sin(angle_rad)

            # Transform to world frame
            world_x = rx + local_x * math.cos(rtheta) - local_y * math.sin(rtheta)
            world_y = ry + local_x * math.sin(rtheta) + local_y * math.cos(rtheta)
            points.append((world_x, world_y))
        
        return points
    
    def segment_scan(self, points, distance_threshold=0.05):
        # Split points into segments based on gaps (walls)
        if len(points) < 2:
            return []
        
        segments = []
        current_segment = [points[0]]

        for i in range(1, len(points)):
            dist = math.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)

            if dist < distance_threshold:
                current_segment.append(points[i])
            else:
                if len(current_segment) >= 3:
                    segments.append(current_segment)
                current_segment = [points[i]]
        
        if len(current_segment) >= 3:
            segments.append(current_segment)
        
        return segments
    
    def fit_line(self, segment):
        # Fit a line to points in a segment of points, return line as point and direction
        points = np.array(segment)
        centroid = np.mean(points, axis=0)
        
        # PCA to find line direction
        centered = points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        direction = eigenvectors[:, np.argmax(eigenvalues)]

        return centroid, direction
    
    def line_intersection(self, p1, d1, p2, d2):
        # Find intersections of two lines defined by point and direction
        # Solving p1 + t*d1 = p2 + s*d2
        A = np.array([[d1[0], -d2[0]], 
                  [d1[1], -d2[1]]])
        b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        
        if abs(np.linalg.det(A)) < 1e-6:
            return None  # parallel lines
        
        t = np.linalg.solve(A, b)[0]
        intersection = p1 + t * d1
        return intersection

    def detect_corners(self, points, distance_threshold=0.15, angle_threshold=10):
        segments = self.segment_scan(points, distance_threshold)
        print(f"  Segments: {len(segments)}, sizes: {[len(s) for s in segments]}")

        if len(segments) < 2:
            return []
        
        lines = []
        for seg in segments:
            centroid, direction = self.fit_line(seg)
            lines.append((centroid, direction, seg))

        corners = []
        # Check ALL pairs of segments, not just adjacent
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p1, d1, seg1 = lines[i]
                p2, d2, seg2 = lines[j]

                angle = math.degrees(math.acos(min(1, abs(np.dot(d1, d2)))))
                if angle < angle_threshold:
                    continue

                intersection = self.line_intersection(p1, d1, p2, d2)
                if intersection is not None:
                    # Check if corner is near either segment's endpoints
                    dists = [
                        math.sqrt((intersection[0]-seg1[0][0])**2 + (intersection[1]-seg1[0][1])**2),
                        math.sqrt((intersection[0]-seg1[-1][0])**2 + (intersection[1]-seg1[-1][1])**2),
                        math.sqrt((intersection[0]-seg2[0][0])**2 + (intersection[1]-seg2[0][1])**2),
                        math.sqrt((intersection[0]-seg2[-1][0])**2 + (intersection[1]-seg2[-1][1])**2),
                    ]
                    if min(dists) < 0.25:
                        corners.append(tuple(intersection))
        
        return corners
    
    def plot_scan(self, points, pose, corners=None):
        # plot scan points and robot
        plt.figure(figsize=(10,10))

        # plot scan points
        if points:
            xs, ys = zip(*points)
            plt.scatter(xs, ys, c='blue', s=20, label='scan points')

        # Plot corners
        if corners:
            cx, cy = zip(*corners)
            plt.scatter(cx, cy, c='green', s=100, marker='s', label='corners')

        # Plot robot
        rx, ry, rtheta = pose
        plt.plot(rx, ry, 'ro', markersize=10, label='Robot')

        # Robot heading arrow
        arrow_len = 0.1
        plt.arrow(rx, ry, arrow_len*math.cos(rtheta), arrow_len*math.sin(rtheta), head_width=0.03, color='red')

        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_map(self, ekf, trajectory=None):
        # plot final map with landmarks and robot trajectory
        plt.figure(figsize=(12,12))

        # Plot trajectory if provided
        if trajectory and len(trajectory) > 1:
            tx, ty = zip(*[(p[0], p[1]) for p in trajectory])
            plt.plot(tx, ty, 'b-', linewidth=2, label='Trajectory')
            plt.plot(tx[0], ty[0], 'go', markersize=12, label='Start')

        # Plot landmarks
        landmarks = ekf.get_landmarks()
        if landmarks:
            lx, ly = zip(*landmarks)
            plt.scatter(lx, ly, c='purple', s=150, marker='^', label=f'Landmarks ({len(landmarks)})')
        
        # Label each landmark
        for i, (x, y) in enumerate(landmarks):
            plt.annotate(f'L{i+1}', (x, y), textcoords='offset points', xytext=(5, 5))
    
        # Plot robot pose
        pose = ekf.get_pose()
        rx, ry, rtheta = pose
        plt.plot(rx, ry, 'ro', markersize=12, label='Robot')
        
        arrow_len = 0.05
        plt.arrow(rx, ry,
                arrow_len * math.cos(rtheta),
                arrow_len * math.sin(rtheta),
                head_width=0.02, color='red')
        
        # Plot robot covariance ellipse
        cov = ekf.P[:2, :2]
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(np.abs(eigenvalues))
        
        ellipse = plt.matplotlib.patches.Ellipse(
            (rx, ry), width, height, angle=angle,
            fill=False, color='red', linestyle='--', linewidth=2
        )
        plt.gca().add_patch(ellipse)
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('EKF-SLAM Map')
        plt.show()

class EKFSlam:
    def __init__(self):
        # State: [x, y, theta, lx1, ly1, ...]
        self.state = np.array([0.0, 0.0, 0.0])

        # Covariance
        self.P = np.diag([0.1, 0.1, 0.1]) # Assume small initial uncertainty
        # Motion noise
        self.Q = np.diag([0.001, 0.001, 0.02]) # x, y, theta variance per motion
        # Observation (measurement) noise
        self.R = np.diag([0.1,0.2]) # range, bearing variance

        # Data association threshold
        self.association_threshold = 0.6 # meters

        # Number of landmarks 
        self.n_landmarks = 0
    
    def predict(self, dx, dy, dtheta):
        # Prediction step using odometry
        # Update robot pose
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = self.normalize_angle(self.state[2])

        # Jacobian of motion model wrt state
        n = len(self.state)
        F = np.eye(n)       # dx and dy are already provided in world frame so F is identity

        # expand Q to full state space
        Q_full = np.zeros((n,n))
        Q_full[:3, :3] = self.Q

        # Update covariance
        self.P = F @ self.P @ F.T + Q_full

    def update(self, observations):
        # Update step using corner observations

        for obs in observations:
            r, b = obs # range and bearing
            # convert observation to world frame
            rx,ry,rtheta = self.state[:3]
            lx = rx + r*np.cos(rtheta+b)
            ly = ry + r*np.sin(rtheta+b)

            # Data association - find nearest landmark
            landmark_idx = self.associate_landmark(lx, ly)

            if landmark_idx is None:
                # new landmark, add
                self.add_landmark(lx, ly)
            else:
                # update landmark if exists
                self.update_landmark(landmark_idx, r, b)
        
    def associate_landmark(self, lx, ly):
        # Find nearest landmark within a threshold or None if new
        if self.n_landmarks == 0:
            return None
        
        # initialize as defaults
        min_dist = float('inf')
        best_idx = None

        # Check all landmarks to see which is closest
        for i in range(self.n_landmarks):
            idx = 3 + i*2
            dx = self.state[idx] - lx
            dy = self.state[idx + 1] - ly
            dist = np.sqrt(dx**2 + dy**2)

            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        print(f"    Observed corner at ({lx:.3f}, {ly:.3f}), nearest landmark L{best_idx+1} at dist {min_dist:.3f}m")
        
        if min_dist < self.association_threshold:
            print(f"    -> Associated with L{best_idx+1}")
            return best_idx
        print(f"    -> New landmark (threshold {self.association_threshold})")
        return None
    
    def add_landmark(self, lx, ly):
        # Add new landmark to state
        # extend state
        self.state = np.append(self.state, [lx, ly])

        # extend covariance
        n = len(self.P)
        P_new = np.zeros((n+2, n+2))
        P_new[:n, :n] = self.P

        # Initial landmark uncertainty (large)
        P_new[n,n] = 0.5
        P_new[n+1, n+1] = 0.5

        # Correlation with robot pose
        # Jacobian of landmark position wrt pose
        rx, ry, rtheta = self.state[:3]
        r = np.sqrt((lx-rx)**2 + (ly-ry)**2)
        b = np.arctan2(ly-ry, lx-rx) - rtheta

        G = np.array([
            [1, 0, -r*np.sin(rtheta +b)],
            [0, 1, r*np.cos(rtheta+b)]
        ])

        P_new[n:n+2, :3] = G @ self.P[:3, :3]
        P_new[:3, n:n+2] = P_new[n:n+2, :3].T

        self.P = P_new
        self.n_landmarks += 1
        print(f"Added Landmark {self.n_landmarks}: ({lx:.3f}, {ly:.3f})")
    
    def update_landmark(self, landmark_idx, r_obs, b_obs):
        # EKF Update for observing existing landmark
        idx = 3 + landmark_idx*2
        lx, ly = self.state[idx], self.state[idx+1]
        rx, ry, rtheta = self.state[:3]

        # Predicted observation
        dx = lx - rx
        dy = ly - ry
        r_pred = np.sqrt(dx**2 + dy**2)
        b_pred = self.normalize_angle(np.arctan2(dy, dx) - rtheta)

        # Innovation
        z = np.array([r_obs - r_pred, self.normalize_angle(b_obs - b_pred)])

        # Innovation gating - reject if too large
        if abs(z[0]) > 0.5 or abs(z[1]) > np.radians(45):
            print(f"    Rejected update: innovation too large (dr={z[0]:.3f}, db={np.degrees(z[1]):.1f}°)")
            return

        # Jacobian of observation model
        n = len(self.state)
        H = np.zeros((2,n))

        # wrt robot pose
        H[0, 0] = -dx / r_pred  # dr/drx
        H[0, 1] = -dy / r_pred  # dr/dry
        H[0, 2] = 0             # dr/dtheta
        H[1, 0] = dy / (r_pred**2)   # db/drx
        H[1, 1] = -dx / (r_pred**2)  # db/dry
        H[1, 2] = -1                  # db/dtheta

        # w.r.t landmark
        H[0, idx] = dx / r_pred      # dr/dlx
        H[0, idx + 1] = dy / r_pred  # dr/dly
        H[1, idx] = -dy / (r_pred**2)     # db/dlx
        H[1, idx + 1] = dx / (r_pred**2)  # db/dly

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ z
        self.state[2] = self.normalize_angle(self.state[2])
        self.P = (np.eye(n) - K @ H) @ self.P

    def normalize_angle(self, angle):
        """Keep angle between -pi and pi"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_pose(self):
        return self.state[:3].copy()
    
    def get_landmarks(self):
        # Return list of landmark positions
        landmarks = []
        for i in range(self.n_landmarks):
            idx = 3 + i*2
            landmarks.append((self.state[idx], self.state[idx+1]))
        return landmarks
    
    def corners_to_observations(self, corners, pose):
        # Convert corner positions to (range, bearing) observations
        rx, ry, rtheta = pose
        observations = []

        for cx, cy in corners:
            dx = cx - rx
            dy = cy - ry
            r = np.sqrt(dx**2 + dy**2)
            b = np.arctan2(dy, dx) - rtheta
            observations.append((r,b))

        return observations

class Explorer:
    def __init__(self, ev3, scans, ekf):
        self.ev3 = ev3
        self.scans = scans
        self.ekf = ekf
        self.visited_position = []
        self.entry_pose = None

    def detect_gaps(self, scan_result, pose, min_gap_width=0.5):
        # Detect gaps in scan that could be doors
        points = self.scans.scan_to_cartesian(scan_result['scan'], pose)
        rx, ry, rtheta = pose
        gaps = []

        if len(points)<2:
            return gaps
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]

            # Distance between consecutive scan points
            gap_width = math.sqrt((p2[0] - p1[0])**2 + (p2[1]-p1[1])**2)

            if gap_width > min_gap_width:
                # Discovered gap - calculate center 
                gap_center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                gap_angle = math.atan2(gap_center[1] - ry, gap_center[0] - rx)
                gap_dist = math.sqrt((gap_center[0] - rx)**2 + (gap_center[1] - ry)**2)

                gaps.append({
                    'center': gap_center,
                    'width': gap_width,
                    'angle': gap_angle,
                    'distance': gap_dist
                })
                print(f"  Gap found: width={gap_width:.2f}m, dist={gap_dist:.2f}m, angle={math.degrees(gap_angle):.1f}°")
        return gaps
    
    def is_near_entry(self, pose, threshold=0.5):
        # Check if robot is near entry point
        if self.entry_pose is None:
            return False
        dx = pose[0] - self.entry_pose[0]
        dy = pose[1] - self.entry_pose[1]
        return math.sqrt(dx**2 + dy**2) < threshold
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def wall_follow(self, side='right', wall_dist=0.15):
        # Follow wall on side given at distance, avoiding obstacles
        self.entry_pose = self.ekf.get_pose().copy()
        steps_since_entry = 0
        Kp = 0.5

        # Follow wall until an exit
        while True:
            pose = self.ekf.get_pose()
            result = self.ev3.scan(step=5)

            if result is None:
                print("Scan error")
                continue

            # Update ekf
            points = self.scans.scan_to_cartesian(result['scan'], pose)
            corners = self.scans.detect_corners(points)
            observations = self.ekf.corners_to_observations(corners, pose)
            self.ekf.update(observations)

            # Check for gaps
            gaps = self.detect_gaps(result, pose)

            for gap in gaps:
                # Skip if near entry - could be entry door
                if steps_since_entry < 10 and self.is_near_entry(pose):
                    print (" Ignoring gap near entry")
                    continue

                # Found potential exit
                if gap['width'] > 0.3 and gap ['distance'] < 1.5:
                    print(f"Exit Found at angle {math.degrees(gap['angle']):.1f}")
                    return gap
            
            # Wall following logic
            front_dist, right_dist, left_dist = self.get_distances(result, pose)

            if front_dist < 0.15:
                # Wall ahead - turn away
                if side == 'right':
                    self.rotate(-45)
                else:
                    self.rotate(45)
            elif side == 'right':
                if abs(right_dist-wall_dist) > 0.1:
                    rot = Kp*(right_dist - wall_dist)
                    self.rotate(rot)
                    self.move(10)
                else:
                    self.move(10)
            else: # left wall
                if abs(left_dist-wall_dist) > 0.1:
                    rot = Kp*(wall_dist - left_dist)
                    self.rotate(rot)
                    self.move(10)
                else:
                    self.move(10)
            
            steps_since_entry += 1
    
    def wall_follow_step(self, scan_result, pose, wall_dist=0.15, side='right', Kp=100):
        """Execute one step of wall following, returns action taken"""
        front_dist, right_dist, left_dist = self.get_distances(scan_result, pose)
        print(f"  F={front_dist:.2f}m, R={right_dist:.2f}m, L={left_dist:.2f}m")
        
        if front_dist < 0.2:
            # Wall ahead - turn away
            if side == 'right':
                print("  Wall ahead, turning left")
                return ('rotate', -45)
            else:
                print("  Wall ahead, turning right")
                return ('rotate', +45)
        elif side == 'right':
            if abs(right_dist - wall_dist) > 0.1:
                rot = Kp * (right_dist - wall_dist)
                print(f"  Adjusting: rotating {rot:.1f}° and moving")
                return ('rotate_move', rot, 10)
            else:
                print("  Following wall")
                return ('move', 10)
        else:  # left wall
            if abs(left_dist - wall_dist) > 0.1:
                rot = Kp * (wall_dist - left_dist)
                print(f"  Adjusting: rotating {rot:.1f}° and moving")
                return ('rotate_move', rot, 10)
            else:
                print("  Following wall")
                return ('move', 10)
    
    def get_distances(self, scan_result, pose):
        # Get distances in front, right, left directions
        points = self.scans.scan_to_cartesian(scan_result['scan'], pose)
        rx, ry, rtheta = pose

        front_dist = 2.0
        right_dist = 2.0
        left_dist = 2.0

        for px, py in points:
            dist = math.sqrt((px-rx)**2 + (py-ry)**2)
            angle = self.normalize_angle(math.atan2(py-ry, px-rx)-rtheta)

            # Front: -45 to +45 degrees
            if abs(angle) < math.radians(45):
                front_dist = min(front_dist, dist)
            # Right: -60 to -120
            elif -math.radians(120) < angle < -math.radians(60):
                right_dist = min(left_dist, dist)
            # Left: 60 to 120
            elif math.radians(60) < angle < math.radians(120):
                left_dist = min(right_dist, dist)
        
        return front_dist, right_dist, left_dist

    def move(self, distance_cm):
        result = self.ev3.move(left_speed=15, right_speed=15, distance_cm=distance_cm)
        if result:
            dx, dy, dtheta = self.scans.odometry(
                result['odometry']['delta_left'],
                result['odometry']['delta_right'],
                self.ekf.get_pose()[2]
            )
            self.ekf.predict(dx, dy, dtheta)
    
    def rotate(self, angle_deg):
        result = self.ev3.rotate(speed=10, angle_deg=angle_deg)
        if result:
            dx, dy, dtheta = self.scans.odometry(
                result['odometry']['delta_left'],
                result['odometry']['delta_right'],
                self.ekf.get_pose()[2]
            )
            self.ekf.predict(dx, dy, dtheta)

    def go_through_gap(self, gap, occ_grid=None, trajectory=None, plotter=None):
        """Navigate through a detected gap while scanning for obstacles"""
        print(f"Navigating to gap at ({gap['center'][0]:.2f}, {gap['center'][1]:.2f})")
        
        target_x, target_y = gap['center']
        max_attempts = 20
        
        for attempt in range(max_attempts):
            pose = self.ekf.get_pose()
            rx, ry, rtheta = pose
            
            # Distance and angle to gap
            dx = target_x - rx
            dy = target_y - ry
            dist_to_gap = math.sqrt(dx**2 + dy**2)
            angle_to_gap = math.atan2(dy, dx)
            angle_diff = self.normalize_angle(angle_to_gap - rtheta)
            
            print(f"  Attempt {attempt+1}: dist={dist_to_gap:.2f}m, angle_diff={math.degrees(angle_diff):.1f}°")
            
            # Check if we've passed through
            if dist_to_gap < 0.15:
                print("  Reached gap center!")
                self.move(25)
                print("Passed through gap!")
                return True
            
            # Scan for obstacles
            result = self.ev3.scan(step=10)
            if result is None:
                print("  Scan error")
                continue
            
            # Update EKF
            points = self.scans.scan_to_cartesian(result['scan'], pose)
            corners = self.scans.detect_corners(points)
            observations = self.ekf.corners_to_observations(corners, pose)
            self.ekf.update(observations)
            
            # Update occupancy grid and plotter if provided
            if occ_grid is not None:
                occ_grid.store_scan(result['scan'], pose)
                occ_grid.update(pose, points)
            if trajectory is not None:
                trajectory.append(pose.copy())
            if plotter is not None:
                plotter.update(self.ekf, trajectory)
            
            # Get distances
            front_dist, right_dist, left_dist = self.get_distances(result, pose)
            print(f"  F={front_dist:.2f}m, R={right_dist:.2f}m, L={left_dist:.2f}m")
            
            # ALWAYS check for obstacle first, before turning or moving
            if front_dist < 0.15:
                print("  Obstacle ahead! Avoiding...")
                if left_dist > right_dist:
                    self.rotate(-30)
                else:
                    self.rotate(30)
                continue  # Re-scan after avoiding
            
            # Now safe to turn or move
            if abs(angle_diff) > math.radians(15):
                turn_angle = -math.degrees(angle_diff)
                print(f"  Turning {turn_angle:.1f}° toward gap")
                self.rotate(turn_angle)
            else:
                # Aligned and clear - move forward
                move_dist = min(15, dist_to_gap * 100)
                move_dist = max(5, move_dist)
                print(f"  Moving {move_dist:.0f}cm toward gap")
                self.move(move_dist)
        
        print("  Max attempts reached")
        return False

class PathPlanner:
    def __init__(self, occ_grid, robot_radius=0.10):
        self.grid = occ_grid
        self.robot_radius = robot_radius
        self.inflated_grid = None
    
    def inflate_obstacles(self):
        """Expand obstacles by robot radius for safe path planning"""
        prob_grid = self.grid.get_probability_grid()
        self.inflated_grid = prob_grid.copy()
        
        # How many cells to inflate
        inflate_cells = int(self.robot_radius / self.grid.resolution) + 1
        
        print(f"  Inflating obstacles by {inflate_cells} cells ({self.robot_radius}m)")
        
        for y in range(self.grid.n_cells):
            for x in range(self.grid.n_cells):
                if prob_grid[y, x] > 0.6:  # Occupied cell
                    # Mark nearby cells as occupied too
                    for dy in range(-inflate_cells, inflate_cells + 1):
                        for dx in range(-inflate_cells, inflate_cells + 1):
                            ny, nx = y + dy, x + dx
                            if self.grid.in_bounds(nx, ny):
                                # Use circular inflation
                                dist = math.sqrt(dx**2 + dy**2)
                                if dist <= inflate_cells:
                                    self.inflated_grid[ny, nx] = max(
                                        self.inflated_grid[ny, nx], 
                                        0.7
                                    )
    
    def find_frontiers(self, pose):
        """Find boundaries between free space and unknown space"""
        prob_grid = self.grid.get_probability_grid()
        frontiers = []
        
        # Free < 0.4, Unknown ~0.5, Occupied > 0.6
        for y in range(1, self.grid.n_cells - 1):
            for x in range(1, self.grid.n_cells - 1):
                if prob_grid[y, x] < 0.4:  # Free cell
                    # Check if adjacent to unknown
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0.45 < prob_grid[ny, nx] < 0.55:  # Unknown
                            wx, wy = self.grid.grid_to_world(x, y)
                            frontiers.append((wx, wy))
                            break
        
        if not frontiers:
            return []
        
        # Cluster nearby frontier cells
        clustered = self.cluster_frontiers(frontiers)
        
        # Sort by distance from robot
        rx, ry = pose[0], pose[1]
        clustered.sort(key=lambda f: math.sqrt((f[0] - rx)**2 + (f[1] - ry)**2))
        
        return clustered
    
    def cluster_frontiers(self, frontiers, cluster_dist=0.1):
        """Group nearby frontier points into clusters, return centers"""
        if not frontiers:
            return []
        
        clusters = []
        used = [False] * len(frontiers)
        
        for i, (fx, fy) in enumerate(frontiers):
            if used[i]:
                continue
            
            cluster = [(fx, fy)]
            used[i] = True
            
            for j, (ox, oy) in enumerate(frontiers):
                if not used[j]:
                    dist = math.sqrt((fx - ox)**2 + (fy - oy)**2)
                    if dist < cluster_dist:
                        cluster.append((ox, oy))
                        used[j] = True
            
            # Cluster center
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            clusters.append((cx, cy, len(cluster)))  # x, y, size
        
        # Filter small clusters and return as (x, y)
        return [(c[0], c[1]) for c in clusters if c[2] >= 3]
    
    def find_gaps_in_grid(self, pose, min_gap_width=0.25):
        """Find gaps in occupied cells that could be doors"""
        prob_grid = self.grid.get_probability_grid()
        rx, ry, rtheta = pose
        gaps = []
        
        # Cast rays from robot position in all directions
        for angle_deg in range(0, 360, 5):
            angle = math.radians(angle_deg)
            
            # Cast ray and record what we hit
            hit_wall = False
            last_free_point = None
            wall_start = None
            wall_end = None
            
            for dist in np.arange(0.1, 1.5, self.grid.resolution):
                wx = rx + dist * math.cos(angle)
                wy = ry + dist * math.sin(angle)
                gx, gy = self.grid.world_to_grid(wx, wy)
                
                if not self.grid.in_bounds(gx, gy):
                    break
                
                prob = prob_grid[gy, gx]
                
                if prob < 0.4:  # Free
                    last_free_point = (wx, wy, dist)
                    if hit_wall:
                        wall_end = dist
                elif prob > 0.6:  # Occupied
                    if not hit_wall:
                        wall_start = dist
                    hit_wall = True
            
            # If ray went through free space without hitting a wall, could be a gap
            if last_free_point is not None and not hit_wall:
                if last_free_point[2] > 0.3:  # At least 30cm of clear path
                    gaps.append({
                        'center': (last_free_point[0], last_free_point[1]),
                        'angle': angle,
                        'distance': last_free_point[2]
                    })
        
        # Cluster nearby gaps
        clustered_gaps = []
        used = [False] * len(gaps)
        
        for i, gap in enumerate(gaps):
            if used[i]:
                continue
            
            cluster = [gap]
            used[i] = True
            
            for j, other in enumerate(gaps):
                if not used[j]:
                    dist = math.sqrt(
                        (gap['center'][0] - other['center'][0])**2 +
                        (gap['center'][1] - other['center'][1])**2
                    )
                    if dist < 0.2:
                        cluster.append(other)
                        used[j] = True
            
            # Average the cluster
            avg_x = sum(g['center'][0] for g in cluster) / len(cluster)
            avg_y = sum(g['center'][1] for g in cluster) / len(cluster)
            avg_dist = sum(g['distance'] for g in cluster) / len(cluster)
            avg_angle = sum(g['angle'] for g in cluster) / len(cluster)
            
            clustered_gaps.append({
                'center': (avg_x, avg_y),
                'angle': avg_angle,
                'distance': avg_dist,
                'size': len(cluster)
            })
        
        # Filter to significant gaps (multiple rays passed through)
        significant = [g for g in clustered_gaps if g['size'] >= 3]
        
        return significant
    
    def astar(self, start_pose, goal, use_inflation=True):
        """A* path planning on occupancy grid"""
        import heapq
        
        # Inflate obstacles if needed
        if use_inflation:
            self.inflate_obstacles()
            grid_to_use = self.inflated_grid
        else:
            grid_to_use = self.grid.get_probability_grid()
        
        start_gx, start_gy = self.grid.world_to_grid(start_pose[0], start_pose[1])
        goal_gx, goal_gy = self.grid.world_to_grid(goal[0], goal[1])
        
        if not self.grid.in_bounds(start_gx, start_gy):
            print("  Start out of bounds")
            return None
        if not self.grid.in_bounds(goal_gx, goal_gy):
            print("  Goal out of bounds")
            return None
        
        # Check if start or goal is in obstacle
        if grid_to_use[start_gy, start_gx] > 0.5:
            print("  Start is in obstacle!")
            return None
        if grid_to_use[goal_gy, goal_gx] > 0.5:
            print("  Goal is in obstacle!")
            return None
        
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def is_valid(x, y):
            if not self.grid.in_bounds(x, y):
                return False
            return grid_to_use[y, x] < 0.5
        
        open_set = [(0, start_gx, start_gy)]
        came_from = {}
        g_score = {(start_gx, start_gy): 0}
        
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            
            if (cx, cy) == (goal_gx, goal_gy):
                # Reconstruct path
                path = []
                current = (goal_gx, goal_gy)
                while current in came_from:
                    wx, wy = self.grid.grid_to_world(current[0], current[1])
                    path.append((wx, wy))
                    current = came_from[current]
                path.reverse()
                return path
            
            # Skip if already processed with better score
            if (cx, cy) in came_from and (cx, cy) != (start_gx, start_gy):
                continue
            
            # 8-connected neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                
                if not is_valid(nx, ny):
                    continue
                
                # Diagonal moves cost more
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[(cx, cy)] + move_cost
                
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), (goal_gx, goal_gy))
                    heapq.heappush(open_set, (f_score, nx, ny))
        
        print("  No path found")
        return None
    
    def simplify_path(self, path, tolerance=0.1):
        """Reduce path to key waypoints"""
        if not path or len(path) < 3:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            curr = path[i]
            next_pt = path[i + 1]
            
            # Check if direction changes significantly
            dir1 = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            dir2 = math.atan2(next_pt[1] - curr[1], next_pt[0] - curr[0])
            
            angle_diff = abs(dir1 - dir2)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            if angle_diff > 0.3:  # ~17 degrees
                simplified.append(curr)
        
        simplified.append(path[-1])
        return simplified
    
    def plot_inflated(self, pose=None, path=None, goal=None, filename=None):
        """Visualize inflated grid with path"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        extent = [-self.grid.size / 2, self.grid.size / 2,
                  -self.grid.size / 2, self.grid.size / 2]
        
        # Original grid
        axes[0].imshow(self.grid.get_probability_grid(), cmap='gray_r',
                       origin='lower', extent=extent, vmin=0, vmax=1)
        axes[0].set_title('Original Grid')
        
        # Inflated grid
        if self.inflated_grid is not None:
            axes[1].imshow(self.inflated_grid, cmap='gray_r',
                           origin='lower', extent=extent, vmin=0, vmax=1)
        else:
            axes[1].imshow(self.grid.get_probability_grid(), cmap='gray_r',
                           origin='lower', extent=extent, vmin=0, vmax=1)
        axes[1].set_title(f'Inflated Grid (r={self.robot_radius}m)')
        
        for ax in axes:
            if pose is not None:
                ax.plot(pose[0], pose[1], 'bo', markersize=12, label='Robot')
                # Draw robot radius circle
                circle = plt.Circle((pose[0], pose[1]), self.robot_radius,
                                     fill=False, color='blue', linestyle='--')
                ax.add_patch(circle)
            
            if goal is not None:
                ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
            
            if path is not None and len(path) > 0:
                px, py = zip(*path)
                ax.plot(px, py, 'g-', linewidth=2, label='Path')
                ax.plot(px[0], py[0], 'go', markersize=8)
                ax.plot(px[-1], py[-1], 'gx', markersize=10)
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"  Saved {filename}")
        plt.show()
    
    def plot_gaps(self, pose, gaps, filename=None):
        """Visualize detected gaps"""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        extent = [-self.grid.size / 2, self.grid.size / 2,
                  -self.grid.size / 2, self.grid.size / 2]
        
        ax.imshow(self.grid.get_probability_grid(), cmap='gray_r',
                  origin='lower', extent=extent, vmin=0, vmax=1)
        
        rx, ry, rtheta = pose
        ax.plot(rx, ry, 'bo', markersize=12, label='Robot')
        
        # Draw robot heading
        ax.arrow(rx, ry, 0.1 * math.cos(rtheta), 0.1 * math.sin(rtheta),
                 head_width=0.03, color='blue')
        
        # Draw gaps
        for i, gap in enumerate(gaps):
            gx, gy = gap['center']
            ax.plot(gx, gy, 'g*', markersize=15)
            ax.plot([rx, gx], [ry, gy], 'g--', linewidth=2)
            ax.annotate(f"Gap {i + 1}\nd={gap['distance']:.2f}m",
                        (gx, gy), textcoords='offset points',
                        xytext=(10, 10), color='green')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Detected Gaps')
        
        if filename:
            plt.savefig(filename)
            print(f"  Saved {filename}")
        plt.show()

class OccupancyGrid:
    def __init__(self, size=5.0, resolution=0.02):
        # Size: total map size in meters
        # Resolution: Cell size in meters
        self.resolution = resolution
        self.size = size
        self.n_cells = int(size/resolution)

        # Grid centered at origin, 0 = unknown, positive = occupied, negative=free
        # Log-odds used for occupancy
        self.grid = np.zeros((self.n_cells, self.n_cells))

        # Log-odds parameters
        self.l_occ = 0.6 # hit
        self.l_free = -0.4 # miss
        self.l_max = 5.0
        self.l_min = -5.0

        self.scan_history = []
    
    def world_to_grid(self,x,y):
        # Convert world coordinates to grid indicie
        gx = int((x+self.size/2)/self.resolution)
        gy = int((y+self.size/2)/self.resolution)
        return gx,gy
    
    def grid_to_world(self,gx,gy):
        # Convert grid indicies to world coordinates
        x = gx*self.resolution - self.size/2
        y = gy*self.resolution - self.size/2
        return x,y
    
    def in_bounds(self,gx,gy):
        return 0 <= gx < self.n_cells and 0 <=gy < self.n_cells
    
    def store_scan(self, raw_scan, pose):
        # Store raw scan data for reconstruction
        self.scan_history.append({
            'raw_scan': [list(reading) for reading in raw_scan], # copy
            'pose': list(pose) #x, y, theta at scan time
        })
    
    def scan_to_world_points(self, raw_scan, pose):
        """Convert raw scan to world coordinates using given pose"""
        rx, ry, rtheta = pose
        points = []
        
        for angle_deg, distance_cm in raw_scan:
            if distance_cm > 150:  # Filter distant readings
                continue
            
            angle_rad = math.radians(-angle_deg)
            local_x = (distance_cm / 100) * math.cos(angle_rad)
            local_y = (distance_cm / 100) * math.sin(angle_rad)
            
            world_x = rx + local_x * math.cos(rtheta) - local_y * math.sin(rtheta)
            world_y = ry + local_x * math.sin(rtheta) + local_y * math.cos(rtheta)
            points.append((world_x, world_y))
        
        return points

    def update(self, pose, points):
        # Update grid with scan data
        rx, ry, _= pose
        robot_gx, robot_gy = self.world_to_grid(rx,ry)

        for px, py in points:
            # mark endpoint as occupied
            gx, gy = self.world_to_grid(px, py)
            if self.in_bounds(gx, gy):
                self.grid[gy,gx] = np.clip(self.grid[gy,gx] + self.l_occ, self.l_min, self.l_max)
            
            # Trace ray from robot to point marka s free
            self.trace_ray(robot_gx, robot_gy, gx, gy)

    def trace_ray(self, x0, y0, x1, y1):
        # Bresenham's line algorithm to mark free cells
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            # Don't mark the endpoint (it's occupied)
            if x == x1 and y == y1:
                break
            
            if self.in_bounds(x, y):
                self.grid[y, x] = np.clip(
                    self.grid[y, x] + self.l_free,
                    self.l_min, self.l_max
                )
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def rebuild(self, pose_corrections=None):
        # Rebuild map from stored scans
        self.grid = np.zeros((self.n_cells, self.n_cells))

        for i, scan_record in enumerate(self.scan_history):
            # use corrected pose if available
            if pose_corrections and i in pose_corrections:
                pose = pose_corrections[i]
            else:
                pose = scan_record['pose']
            
            # Convert raw scan to world points
            points = self.scan_to_world_points(scan_record['raw_scan'], pose)

            # Update grid
            self.update(pose, points)

    def rebuild_with_ekf(self, trajectory):
        if len(trajectory) != len(self.scan_history):
            print(f"Warning: trajectory length ({len(trajectory)}) != scan history ({len(self.scan_history)})")
            # Use minimum of both
            n = min(len(trajectory), len(self.scan_history))
        else:
            n = len(trajectory)
                
        self.grid = np.zeros((self.n_cells, self.n_cells))
        
        for i in range(n):
            pose = trajectory[i]
            raw_scan = self.scan_history[i]['raw_scan']
            points = self.scan_to_world_points(raw_scan, pose)
            self.update(pose, points)
        
        print("Map rebuild complete")

    def get_probability_grid(self):
        # convert log odds to probability
        return 1 - 1/(1 + np.exp(self.grid))
    
    def plot(self, ekf=None, trajectory=None):
        # plot occupancy grid with optional ekf overlay
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot occupancy grid
        prob_grid = self.get_probability_grid()
        extent = [-self.size/2, self.size/2, -self.size/2, self.size/2]
        ax.imshow(prob_grid, cmap='gray_r', origin='lower', 
                  extent=extent, vmin=0, vmax=1)
        
        if trajectory and len(trajectory) > 1:
            tx, ty = zip(*[(p[0], p[1]) for p in trajectory])
            ax.plot(tx, ty, 'b-', linewidth=2, label='Trajectory')
            ax.plot(tx[0], ty[0], 'go', markersize=10, label='Start')
        
        if ekf is not None:
            # Plot landmarks
            landmarks = ekf.get_landmarks()
            if landmarks:
                lx, ly = zip(*landmarks)
                ax.scatter(lx, ly, c='red', s=100, marker='^', 
                          label=f'Landmarks ({len(landmarks)})', zorder=5)
                for i, (x, y) in enumerate(landmarks):
                    ax.annotate(f'L{i+1}', (x, y), textcoords='offset points', 
                               xytext=(5, 5), color='red')
            
            # Plot robot
            pose = ekf.get_pose()
            rx, ry, rtheta = pose
            ax.plot(rx, ry, 'bo', markersize=12, label='Robot', zorder=6)
            arrow_len = 0.1
            ax.arrow(rx, ry,
                    arrow_len * np.cos(rtheta),
                    arrow_len * np.sin(rtheta),
                    head_width=0.03, color='blue', zorder=6)
        
        ax.set_xlim(-self.size/2, self.size/2)
        ax.set_ylim(-self.size/2, self.size/2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Occupancy Grid Map')
        plt.show()
        
class LivePlotter:
    def __init__(self, occ_grid):
        plt.ion() #interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.occ_grid=occ_grid

        # Initial empty plot
        self.img = self.ax.imshow(
            occ_grid.get_probability_grid(),
            cmap='gray_r', origin='lower',
            extent=[-occ_grid.size/2, occ_grid.size/2,
                    -occ_grid.size/2, occ_grid.size/2],
            vmin=0, vmax=1
        )

        self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.robot_dot, = self.ax.plot([], [], 'bo', markersize=12)
        self.landmark_scatter = self.ax.scatter([], [], c='red', s=100, marker='^')

        self.ax.set_xlim(-occ_grid.size/2, occ_grid.size/2)
        self.ax.set_ylim(-occ_grid.size/2, occ_grid.size/2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Live EKF-SLAM')

        plt.show()

    def update(self, ekf, trajectory):
        # update occupancy grid image
        self.img.set_data(self.occ_grid.get_probability_grid())

        # Update trajectory
        if len(trajectory) > 0:
            tx = [p[0] for p in trajectory]
            ty = [p[1] for p in trajectory]
            self.trajectory_line.set_data(tx,ty)

        # Update robot position
        pose = ekf.get_pose()
        self.robot_dot.set_data([pose[0]], [pose[1]])

        # Update landmarks
        landmarks = ekf.get_landmarks()
        if landmarks:
            lx, ly = zip(*landmarks)
            self.landmark_scatter.set_offsets(list(zip(lx,ly)))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.close()

def main():
    ev3 = EV3Connection("/dev/rfcomm0")
    scans = Scans()
    ekf = EKFSlam()
    pose = [0.0, 0.0, 0.0]

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


if __name__ == "__main__":
    main()