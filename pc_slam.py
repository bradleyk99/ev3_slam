#!/usr/bin/env python3
import json
import termios
import time
import math
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
    
    def scan(self, start_angle=-90, end_angle=90, step=10):
        self.send({
            "type": "scan",
            "start_angle": start_angle,
            "end_angle": end_angle,
            "step": step
        })
        return self.recv()
    
    def move_and_scan(self, left_speed, right_speed, distance_cm,
                      start_angle=-90, end_angle=90, step=10):
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
        theta = theta + dtheta

        dx = dist_center * math.cos(theta)
        dy = dist_center * math.sin(theta)

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
            if distance_cm > 200:
                continue

            # Convert to robot frame
            angle_rad = math.radians(angle_deg)
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

    def detect_corners(self, points, distance_threshold=0.08, angle_threshold=20):
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
                    if min(dists) < 0.15:
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
        self.Q = np.diag([0.01, 0.01, 0.1]) # x, y, theta variance per motion
        # Observation (measurement) noise
        self.R = np.diag([0.02,0.05]) # range, bearing variance

        # Data association threshold
        self.association_threshold = 0.2 # meters

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
        
        if min_dist < self.association_threshold:
            return best_idx
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