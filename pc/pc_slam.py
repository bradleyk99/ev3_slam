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
        self.WHEEL_BASE = 0.1 # meters
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
        max_range_angles = []

        for angle_deg, distance_cm in scan:
            angle_rad = math.radians(-angle_deg)

            if distance_cm > 100:   # Values above a threshold detect empty
                max_range_angles.append(angle_rad)
                continue

            # Convert to robot frame
            local_x = (distance_cm/100) * math.cos(angle_rad)
            local_y = (distance_cm/100) * math.sin(angle_rad)

            # Transform to world frame
            world_x = rx + local_x * math.cos(rtheta) - local_y * math.sin(rtheta)
            world_y = ry + local_x * math.sin(rtheta) + local_y * math.cos(rtheta)
            points.append((world_x, world_y))
        
        return points, max_range_angles
    
    def segment_scan(self, points, distance_threshold=0.08, angle_threshold=20):
        """
        Split points into segments based on gaps AND direction changes
        
        Args:
            distance_threshold: Max distance between consecutive points (meters)
            angle_threshold: Max angle change between consecutive point directions (degrees)
        """
        if len(points) < 2:
            return []
        
        segments = []
        current_segment = [points[0]]

        for i in range(1, len(points)):
            # Distance between consecutive points
            dist = math.sqrt((points[i][0] - points[i-1][0])**2 + 
                            (points[i][1] - points[i-1][1])**2)
            
            # Direction change check (if we have enough points)
            angle_change = 0
            if len(current_segment) >= 2:
                # Vector from second-to-last to last point in current segment
                v1_x = current_segment[-1][0] - current_segment[-2][0]
                v1_y = current_segment[-1][1] - current_segment[-2][1]
                
                # Vector from last point to new point
                v2_x = points[i][0] - current_segment[-1][0]
                v2_y = points[i][1] - current_segment[-1][1]
                
                # Angle between vectors
                norm1 = math.sqrt(v1_x**2 + v1_y**2)
                norm2 = math.sqrt(v2_x**2 + v2_y**2)
                
                if norm1 > 1e-6 and norm2 > 1e-6:
                    dot = (v1_x * v2_x + v1_y * v2_y) / (norm1 * norm2)
                    dot = max(-1, min(1, dot))  # Clamp to [-1, 1]
                    angle_change = math.degrees(math.acos(dot))

            # Break segment if distance too large OR direction changed too much
            if dist > distance_threshold or angle_change > angle_threshold:
                if len(current_segment) >= 3:
                    segments.append(current_segment)
                current_segment = [points[i]]
            else:
                current_segment.append(points[i])
        
        if len(current_segment) >= 3:
            segments.append(current_segment)
        
        return segments
    
    def fit_line(self, segment, max_iterations=100, threshold=0.03):
        """Fit a line to points using RANSAC - more robust than PCA"""
        if len(segment) < 2:
            return None, None
        
        points = np.array(segment)
        best_inliers = []
        best_model = None
        
        for _ in range(max_iterations):
            # Randomly sample 2 points
            if len(points) < 2:
                break
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx]
            
            # Fit line through these points
            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-6:
                continue
            direction = direction / direction_norm
            
            # Count inliers (points close to this line)
            inliers = []
            for i, p in enumerate(points):
                # Distance from point to line
                v = p - p1
                projection = np.dot(v, direction) * direction
                perpendicular = v - projection
                dist = np.linalg.norm(perpendicular)
                
                if dist < threshold:
                    inliers.append(i)
            
            # Keep best model
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (p1, direction)
        
        if not best_inliers or len(best_inliers) < 2:
            return None, None
        
        # Refit line using all inliers for better accuracy
        inlier_points = points[best_inliers]
        centroid = np.mean(inlier_points, axis=0)
        
        # PCA on inliers only
        centered = inlier_points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        return centroid, direction
    
    def line_intersection(self, p1, d1, p2, d2):
        """Find intersection of two lines defined by point and direction"""
        # Solving p1 + t*d1 = p2 + s*d2
        A = np.array([[d1[0], -d2[0]], 
                    [d1[1], -d2[1]]])
        b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        
        if abs(np.linalg.det(A)) < 1e-6:
            return None  # parallel lines
        
        t = np.linalg.solve(A, b)[0]
        intersection = p1 + t * d1
        return intersection

    def detect_corners(self, points, distance_threshold=0.2, angle_threshold=45):
        """Enhanced corner detection using RANSAC for line fitting"""
        segments = self.segment_scan(points, distance_threshold, angle_threshold=45)
        print(f"  Segments: {len(segments)}, sizes: {[len(s) for s in segments]}")

        if len(segments) < 2:
            return []
        
        # Fit lines to segments using RANSAC
        lines = []
        for seg in segments:
            if len(seg) < 3:  # Need minimum points for RANSAC
                continue
            
            centroid, direction = self.fit_line(seg, max_iterations=50, threshold=0.02)
            if centroid is not None and direction is not None:
                lines.append((centroid, direction, seg))
        
        if len(lines) < 2:
            return []
        
        corners = []
        
        # Check ALL pairs of segments for corners
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p1, d1, seg1 = lines[i]
                p2, d2, seg2 = lines[j]

                # Check angle between lines
                angle = math.degrees(math.acos(min(1, abs(np.dot(d1, d2)))))
                if angle < angle_threshold:
                    continue

                intersection = self.line_intersection(p1, d1, p2, d2)
                if intersection is None:
                    continue
                
                # Check if corner is near either segment's endpoints
                dists = [
                    math.sqrt((intersection[0]-seg1[0][0])**2 + (intersection[1]-seg1[0][1])**2),
                    math.sqrt((intersection[0]-seg1[-1][0])**2 + (intersection[1]-seg1[-1][1])**2),
                    math.sqrt((intersection[0]-seg2[0][0])**2 + (intersection[1]-seg2[0][1])**2),
                    math.sqrt((intersection[0]-seg2[-1][0])**2 + (intersection[1]-seg2[-1][1])**2),
                ]
                
                # Must be near at least one endpoint of each segment
                near_seg1 = min(dists[0], dists[1]) < 0.2
                near_seg2 = min(dists[2], dists[3]) < 0.2
                
                if near_seg1 and near_seg2:
                    corners.append(tuple(intersection))
        
        # Remove duplicate corners
        return self.remove_duplicate_corners(corners, threshold=0.1)

    def remove_duplicate_corners(self, corners, threshold=0.1):
        """Cluster nearby corners and return unique ones"""
        if not corners:
            return []
        
        unique = []
        used = [False] * len(corners)
        
        for i, c1 in enumerate(corners):
            if used[i]:
                continue
            
            cluster = [c1]
            used[i] = True
            
            for j, c2 in enumerate(corners):
                if not used[j]:
                    dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    if dist < threshold:
                        cluster.append(c2)
                        used[j] = True
            
            # Average the cluster
            avg_x = sum(c[0] for c in cluster) / len(cluster)
            avg_y = sum(c[1] for c in cluster) / len(cluster)
            unique.append((avg_x, avg_y))
        
        return unique
    
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

    def plot_segments_and_lines(self, points, pose):
        """Debug visualization for segmentation and line fitting"""
        segments = self.segment_scan(points)
        
        plt.figure(figsize=(12, 12))
        
        # Plot all points
        if points:
            xs, ys = zip(*points)
            plt.scatter(xs, ys, c='lightblue', s=30, label='Scan points', alpha=0.5)
        
        # Plot segments in different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
        for i, seg in enumerate(segments):
            if len(seg) > 0:
                sx, sy = zip(*seg)
                plt.scatter(sx, sy, c=[colors[i]], s=50, label=f'Seg {i} ({len(seg)} pts)')
                
                # Fit and plot line
                if len(seg) >= 3:
                    centroid, direction = self.fit_line(seg, max_iterations=50, threshold=0.03)
                    if centroid is not None:
                        # Draw line extending through segment
                        t_vals = np.linspace(-0.5, 0.5, 100)
                        line_x = centroid[0] + t_vals * direction[0]
                        line_y = centroid[1] + t_vals * direction[1]
                        plt.plot(line_x, line_y, c=colors[i], linewidth=2, linestyle='--')
        
        # Plot robot
        rx, ry, rtheta = pose
        plt.plot(rx, ry, 'ro', markersize=12, label='Robot')
        plt.arrow(rx, ry, 0.1*math.cos(rtheta), 0.1*math.sin(rtheta), 
                head_width=0.03, color='red')
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Segment Detection and Line Fitting')
        plt.show()

class EKFSlam:
    def __init__(self):
        # State: [x, y, theta, lx1, ly1, ...]
        self.state = np.array([0.0, 0.0, 0.0])

        # Covariance
        self.P = np.diag([0.1, 0.1, 0.1]) # Assume small initial uncertainty
        # Motion noise
        self.Q = np.diag([0.02, 0.02, 0.1]) # x, y, theta variance per motion
        # Observation (measurement) noise
        self.R = np.diag([0.2,0.4]) # range, bearing variance

        # Data association threshold
        self.association_threshold = 0.2# meters

        # Number of landmarks 
        self.n_landmarks = 0
    
    def predict(self, dx, dy, dtheta):
        # Update robot pose
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = self.normalize_angle(self.state[2])

        # Jacobian
        n = len(self.state)
        F = np.eye(n)

        # Expand Q
        Q_full = np.zeros((n, n))
        Q_full[:3, :3] = self.Q

        # Update covariance
        self.P = F @ self.P @ F.T + Q_full
        
        # ENFORCE SYMMETRY
        self.P = (self.P + self.P.T) / 2

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
        """Find nearest landmark using Mahalanobis distance with Euclidean fallback"""
        if self.n_landmarks == 0:
            return None
        
        rx, ry, rtheta = self.state[:3]
        min_mahal_dist = float('inf')
        min_euclidean_dist = float('inf')
        best_idx = None

        for i in range(self.n_landmarks):
            idx = 3 + i*2
            lx_est = self.state[idx]
            ly_est = self.state[idx + 1]
            
            # Euclidean distance (ground truth proximity)
            euclidean_dist = np.sqrt((lx - lx_est)**2 + (ly - ly_est)**2)
            
            # Predicted observation
            dx_pred = lx_est - rx
            dy_pred = ly_est - ry
            r_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            
            if r_pred < 1e-6:
                continue
                
            b_pred = np.arctan2(dy_pred, dx_pred) - rtheta
            b_pred = self.normalize_angle(b_pred)
            
            # Actual observation
            dx_obs = lx - rx
            dy_obs = ly - ry
            r_obs = np.sqrt(dx_obs**2 + dy_obs**2)
            b_obs = np.arctan2(dy_obs, dx_obs) - rtheta
            b_obs = self.normalize_angle(b_obs)
            
            # Innovation in observation space
            innovation = np.array([r_obs - r_pred, self.normalize_angle(b_obs - b_pred)])
            
            n = len(self.state)
            H = np.zeros((2, n))
            
            # Jacobian w.r.t robot pose
            H[0, 0] = -dx_pred / r_pred
            H[0, 1] = -dy_pred / r_pred
            H[0, 2] = 0
            H[1, 0] = dy_pred / (r_pred**2)
            H[1, 1] = -dx_pred / (r_pred**2)
            H[1, 2] = -1
            
            # Jacobian w.r.t landmark
            H[0, idx] = dx_pred / r_pred
            H[0, idx+1] = dy_pred / r_pred
            H[1, idx] = -dy_pred / (r_pred**2)
            H[1, idx+1] = dx_pred / (r_pred**2)
            
            # Innovation covariance
            S = H @ self.P @ H.T + self.R
            
            # Mahalanobis distance
            try:
                # Ensure S is symmetric
                S = (S + S.T) / 2
                S_inv = np.linalg.inv(S)
                mahal_sq = innovation.T @ S_inv @ innovation
                
                if mahal_sq < 0:
                    mahal_dist = 999  # Invalid, will use Euclidean
                else:
                    mahal_dist = np.sqrt(mahal_sq)
                    
            except np.linalg.LinAlgError:
                mahal_dist = 999  # Invalid, will use Euclidean
            
            # Track best by both metrics
            if mahal_dist < min_mahal_dist:
                min_mahal_dist = mahal_dist
                best_idx = i
            if euclidean_dist < min_euclidean_dist:
                min_euclidean_dist = euclidean_dist
        
        # Get the best candidate's Euclidean distance
        best_euclidean = np.sqrt((lx - self.state[3 + best_idx*2])**2 + 
                                (ly - self.state[3 + best_idx*2 + 1])**2)
        
        print(f"    Observed corner at ({lx:.3f}, {ly:.3f}), nearest landmark L{best_idx+1}")
        print(f"      Euclidean dist: {best_euclidean:.3f}m, Mahalanobis dist: {min_mahal_dist:.3f}")
        
        # Two-stage decision:
        # 1. If Mahalanobis is good, associate
        if min_mahal_dist < 1.0:
            print(f"    -> Associated with L{best_idx+1} (Mahalanobis < 1.0)")
            return best_idx
        
        # 2. If Mahalanobis failed BUT Euclidean is very small, still associate
        # This catches cases where EKF uncertainty is wrong
        euclidean_threshold = 0.20  # 20cm - definitely same landmark
        if best_euclidean < euclidean_threshold:
            print(f"    -> Associated with L{best_idx+1} (Euclidean < {euclidean_threshold}m override)")
            return best_idx
        
        # Otherwise, new landmark
        print(f"     -> New landmark (Mahal={min_mahal_dist:.2f} > 1.0, Eucl={best_euclidean:.2f} > {euclidean_threshold})")
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

        # Innovation gating
        if abs(z[0]) > 0.5 or abs(z[1]) > np.radians(90):
            print(f"    Rejected update: innovation too large (dr={z[0]:.3f}, db={np.degrees(z[1]):.1f}°)")
            return

        # Jacobian
        n = len(self.state)
        H = np.zeros((2, n))
        H[0, 0] = -dx / r_pred
        H[0, 1] = -dy / r_pred
        H[0, 2] = 0
        H[1, 0] = dy / (r_pred**2)
        H[1, 1] = -dx / (r_pred**2)
        H[1, 2] = -1
        H[0, idx] = dx / r_pred
        H[0, idx+1] = dy / r_pred
        H[1, idx] = -dy / (r_pred**2)
        H[1, idx+1] = dx / (r_pred**2)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ z
        self.state[2] = self.normalize_angle(self.state[2])
        
        # Update covariance using JOSEPH FORM (numerically stable)
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # ENFORCE SYMMETRY
        self.P = (self.P + self.P.T) / 2

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

            if r > 1.5: # Reject corners detected further than the reliable distance
                print(f"Rejecting corner at ({cx:.2f}, {cy:.2f}) - Too far r = {r:.2f}")

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
        points, _ = self.scans.scan_to_cartesian(scan_result['scan'], pose)
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
            points, _ = self.scans.scan_to_cartesian(result['scan'], pose)
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
    
    def wall_follow_step(self, scan_result, pose, wall_dist=0.15, side='right', Kp=100, dist=10):
        """Execute one step of wall following, returns action taken"""
        front_dist, right_dist, left_dist = self.get_distances(scan_result, pose)
        print(f"  F={front_dist:.2f}m, R={right_dist:.2f}m, L={left_dist:.2f}m")
        
        if front_dist < (dist*2/100):
            # Wall ahead - turn away
            if side == 'right':
                print("  Wall ahead, turning left")
                return ('rotate_move', -45, -1*dist/2)
            else:
                print("  Wall ahead, turning right")
                return ('rotate_move', +45, -1*dist/2)
            
            
        elif side == 'right':
            if abs(right_dist - wall_dist) > 0.05:
                rot = Kp * (right_dist - wall_dist)
                if abs(rot > 90):
                    rot = np.sign(rot)*90
                print(f"  Adjusting: rotating {rot:.1f}° and moving")
                return('rotate_move', rot, dist)
            else:
                print("  Following wall")
                return ('move', dist)
        else:  # left wall
            if abs(left_dist - wall_dist) > 0.05:
                rot = Kp * (wall_dist - left_dist)
                if abs(rot > 90):
                    rot = np.sign(rot)*90
                print(f"  Adjusting: rotating {rot:.1f}° and moving")
                return ('rotate_move', rot, dist)
            else:
                print("  Following wall")
                return ('move', dist)
    
    def get_distances(self, scan_result, pose):
        # Get distances in front, right, left directions
        points, _ = self.scans.scan_to_cartesian(scan_result['scan'], pose)
        rx, ry, rtheta = pose

        front_dist = 2.0
        right_dist = 2.0
        left_dist = 2.0

        for px, py in points:
            dist = math.sqrt((px-rx)**2 + (py-ry)**2)
            angle = self.normalize_angle(math.atan2(py-ry, px-rx)-rtheta)

            # Front: -30 to +30 degrees
            if abs(angle) < math.radians(30):
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
            points, _ = self.scans.scan_to_cartesian(result['scan'], pose)
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
    
    def find_gaps_in_grid(self, pose, min_gap_width=0.20):
        #Find openings in walls - both narrow gaps and wide exits
        prob_grid = self.grid.get_probability_grid()
        rx, ry, rtheta = pose
        
        print(f"\n  === Gap Detection Debug ===")
        print(f"  Robot at ({rx:.2f}, {ry:.2f})")
        
        # Cast rays in all directions
        ray_results = []
        for angle_deg in range(0, 360, 3):
            angle = math.radians(angle_deg)
            wall_dist = None
            
            for dist in np.arange(0.1, 1.5, self.grid.resolution):
                wx = rx + dist * math.cos(angle)
                wy = ry + dist * math.sin(angle)
                gx, gy = self.grid.world_to_grid(wx, wy)
                
                if not self.grid.in_bounds(gx, gy):
                    break
                
                prob = prob_grid[gy, gx]
                
                if prob > 0.6:  # Hit wall
                    wall_dist = dist
                    break
            
            ray_results.append({
                'angle': angle,
                'angle_deg': angle_deg,
                'wall_dist': wall_dist
            })
        
        print(f"  Total rays: {len(ray_results)}")
        rays_no_wall = sum(1 for r in ray_results if r['wall_dist'] is None or r['wall_dist'] > 1.2)
        print(f"  Rays with no close wall: {rays_no_wall}")
        
        # Find sequences of rays with no walls (wide openings)
        gaps = []
        n_rays = len(ray_results)
        
        in_opening = False
        opening_start = 0
        opening_rays = []
        
        # Detect continuous sequences of rays with no walls
        for i in range(n_rays + n_rays // 2):  # Loop around to catch wraparound
            idx = i % n_rays
            ray = ray_results[idx]
            
            is_open = ray['wall_dist'] is None or ray['wall_dist'] > 1.0
            
            if is_open:
                if not in_opening:
                    # Start of new opening
                    in_opening = True
                    opening_start = idx
                    opening_rays = [ray]
                else:
                    # Continue current opening
                    opening_rays.append(ray)
            else:
                if in_opening:
                    # End of opening - process it
                    if len(opening_rays) >= 3:  # At least 9 degrees wide
                        # Find center ray of opening
                        center_idx = len(opening_rays) // 2
                        center_ray = opening_rays[center_idx]
                        
                        # Place gap at fixed distance into opening
                        gap_dist = 0.6
                        gap_x = rx + gap_dist * math.cos(center_ray['angle'])
                        gap_y = ry + gap_dist * math.sin(center_ray['angle'])
                        
                        # Verify location is free
                        gx, gy = self.grid.world_to_grid(gap_x, gap_y)
                        if self.grid.in_bounds(gx, gy) and prob_grid[gy, gx] < 0.5:
                            gaps.append({
                                'center': (gap_x, gap_y),
                                'angle': center_ray['angle'],
                                'distance': gap_dist,
                                'width_deg': len(opening_rays) * 3,  # Angular width
                                'size': len(opening_rays)
                            })
                            print(f"    Found opening: {len(opening_rays)} rays ({len(opening_rays)*3}°) at {math.degrees(center_ray['angle']):.0f}°")
                    
                    in_opening = False
                    opening_rays = []
        
        print(f"  Raw openings found: {len(gaps)}")
        
        # Remove duplicates (openings detected from wraparound)
        unique_gaps = []
        used = [False] * len(gaps)
        
        for i, gap in enumerate(gaps):
            if used[i]:
                continue
            
            # Check if similar gap already added
            is_duplicate = False
            for existing in unique_gaps:
                angle_diff = abs(gap['angle'] - existing['angle'])
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                if angle_diff < math.radians(20):  # Within 20 degrees
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_gaps.append(gap)
        
        print(f"  After deduplication: {len(unique_gaps)} openings")
        
        for i, g in enumerate(unique_gaps):
            print(f"    Opening {i+1}: center=({g['center'][0]:.2f}, {g['center'][1]:.2f}), "
                f"dist={g['distance']:.2f}m, angle={math.degrees(g['angle']):.0f}°, width={g['width_deg']}°")
        
        return unique_gaps
    
    def astar(self, start_pose, goal, use_inflation=True):
        # A* path planning with extensive debug
        import heapq
        
        print(f"\n  === A* Planning Debug ===")
        
        # Inflate obstacles
        if use_inflation:
            self.inflate_obstacles()
            grid_to_use = self.inflated_grid
            print(f"  Using inflated grid (robot radius={self.robot_radius}m)")
        else:
            grid_to_use = self.grid.get_probability_grid()
            print(f"  Using original grid")
        
        start_gx, start_gy = self.grid.world_to_grid(start_pose[0], start_pose[1])
        goal_gx, goal_gy = self.grid.world_to_grid(goal[0], goal[1])
        
        print(f"  Start: world=({start_pose[0]:.2f}, {start_pose[1]:.2f}) -> grid=({start_gx}, {start_gy})")
        print(f"  Goal:  world=({goal[0]:.2f}, {goal[1]:.2f}) -> grid=({goal_gx}, {goal_gy})")
        
        if not self.grid.in_bounds(start_gx, start_gy):
            print(f"  ERROR: Start out of bounds!")
            return None
        if not self.grid.in_bounds(goal_gx, goal_gy):
            print(f"  ERROR: Goal out of bounds!")
            return None
        
        start_prob = grid_to_use[start_gy, start_gx]
        goal_prob = grid_to_use[goal_gy, goal_gx]
        print(f"  Start cell probability: {start_prob:.3f} ({'OCCUPIED' if start_prob > 0.5 else 'FREE'})")
        print(f"  Goal cell probability: {goal_prob:.3f} ({'OCCUPIED' if goal_prob > 0.5 else 'FREE'})")
        
        # Fix start if needed
        if start_prob > 0.5:
            print(f"  WARNING: Start in obstacle, searching for free cell...")
            found = False
            for radius in range(1, 20):
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = start_gx + dx, start_gy + dy
                        if self.grid.in_bounds(nx, ny) and grid_to_use[ny, nx] < 0.5:
                            start_gx, start_gy = nx, ny
                            wx, wy = self.grid.grid_to_world(start_gx, start_gy)
                            print(f"  Moved start to: grid=({start_gx}, {start_gy}), world=({wx:.2f}, {wy:.2f})")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                print(f"  ERROR: No free cells near start!")
                return None
        
        # Fix goal if needed
        if goal_prob > 0.5:
            print(f"  WARNING: Goal in obstacle, searching for free cell...")
            found = False
            for radius in range(1, 30):
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = goal_gx + dx, goal_gy + dy
                        if self.grid.in_bounds(nx, ny) and grid_to_use[ny, nx] < 0.5:
                            goal_gx, goal_gy = nx, ny
                            wx, wy = self.grid.grid_to_world(goal_gx, goal_gy)
                            print(f"  Moved goal to: grid=({goal_gx}, {goal_gy}), world=({wx:.2f}, {wy:.2f})")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                print(f"  ERROR: No free cells near goal!")
                return None
        
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def is_valid(x, y):
            if not self.grid.in_bounds(x, y):
                return False
            return grid_to_use[y, x] < 0.5  # Free if prob < 0.5
        
        print(f"  Starting A* search...")
        open_set = [(0, start_gx, start_gy)]
        came_from = {}
        g_score = {(start_gx, start_gy): 0}
        closed_set = set()
        nodes_explored = 0
        
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            
            # Skip if already processed
            if (cx, cy) in closed_set:
                continue
            
            closed_set.add((cx, cy))
            nodes_explored += 1
            
            if nodes_explored % 100 == 0:
                print(f"  ... explored {nodes_explored} nodes, open_set size={len(open_set)}")
            
            if (cx, cy) == (goal_gx, goal_gy):
                print(f"  SUCCESS! Found path after exploring {nodes_explored} nodes")
                # Reconstruct path
                path = []
                current = (goal_gx, goal_gy)
                while current in came_from:
                    wx, wy = self.grid.grid_to_world(current[0], current[1])
                    path.append((wx, wy))
                    current = came_from[current]
                path.reverse()
                print(f"  Path has {len(path)} waypoints")
                return path
            
            # 8-connected neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                
                if not is_valid(nx, ny):
                    continue
                
                if (nx, ny) in closed_set:
                    continue
                
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[(cx, cy)] + move_cost
                
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), (goal_gx, goal_gy))
                    heapq.heappush(open_set, (f_score, nx, ny))
        
        print(f"  FAILURE: No path found after exploring {nodes_explored} nodes")
        print(f"  Closed set size: {len(closed_set)}")
        
        # Check if goal was reachable
        if (goal_gx, goal_gy) in closed_set:
            print(f"  ERROR: Goal was explored but not reached - logic error!")
        else:
            print(f"  Goal never reached - isolated by obstacles")
            # Check neighbors of goal
            free_neighbors = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = goal_gx + dx, goal_gy + dy
                if self.grid.in_bounds(nx, ny) and grid_to_use[ny, nx] < 0.5:
                    free_neighbors += 1
            print(f"  Goal has {free_neighbors}/4 free neighbors")
        
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

    def update(self, pose, points, max_range_angles=None, max_range=1.0):
        # Update grid with scan data
        rx, ry, rtheta= pose
        robot_gx, robot_gy = self.world_to_grid(rx,ry)

        for px, py in points:
            # mark endpoint as occupied
            gx, gy = self.world_to_grid(px, py)
            if self.in_bounds(gx, gy):
                self.grid[gy,gx] = np.clip(self.grid[gy,gx] + self.l_occ, self.l_min, self.l_max)
            
            # Trace ray from robot to point marka s free
            self.trace_ray(robot_gx, robot_gy, gx, gy)
        
        if max_range_angles is not None:
            for angle in max_range_angles:
                # Project ray to max range
                world_angle=rtheta+angle
                end_x = rx + max_range *math.cos(world_angle)
                end_y = ry + max_range *math.sin(world_angle)

                end_gx, end_gy = self.world_to_grid(end_x,end_y)

                # Only mark ray as free not occupied
                self.trace_ray(robot_gx, robot_gy, end_gx, end_gy)

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
        self.landmark_ellipses = [] # Store ellipse patches
        self.scan_points_scatter = self.ax.scatter([], [], c='cyan', s=20, alpha=0.6)
        self.fitted_lines = []

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

    def update(self, ekf, trajectory, current_scan_points=None, fitted_line_segments=None):
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

        # Update scan points
        if current_scan_points and len(current_scan_points) > 0:
            sx, sy = zip(*current_scan_points)
            self.scan_points_scatter.set_offsets(list(zip(sx, sy)))
        else:
            self.scan_points_scatter.set_offsets(np.empty((0, 2)))
        
        # Update fitted lines
        # Remove old line plots
        for line in self.fitted_lines:
            line.remove()
        self.fitted_lines = []

        if fitted_line_segments:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(fitted_line_segments)))
            for i, (centroid, direction) in enumerate(fitted_line_segments):
                t_vals = np.linspace(-0.3, 0.3, 50)
                line_x = centroid[0] + t_vals * direction[0]
                line_y = centroid[1] + t_vals * direction[1]
                line_plot, = self.ax.plot(line_x, line_y, color=colors[i], 
                                        linewidth=2, linestyle='--', alpha=0.7)
                self.fitted_lines.append(line_plot)
        
        # Update landmark uncertainty ellipses
        # Remove old ellipses
        for ellipse in self.landmark_ellipses:
                ellipse.remove()
        self.landmark_ellipses = []

        # Draw new ellipses
        for i in range(ekf.n_landmarks):
            idx = 3 + i*2
            lx, ly = ekf.state[idx], ekf.state[idx+1]
        
            # Get landmark covariance (2x2)
            P_landmark = ekf.P[idx:idx+2, idx:idx+2]
            
            # Compute ellipse from covariance
            eigenvalues, eigenvectors = np.linalg.eig(P_landmark)
            angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # 1-sigma ellipse (68% confidence)
            width = 2 * np.sqrt(abs(eigenvalues[0]))
            height = 2 * np.sqrt(abs(eigenvalues[1]))
            
            ellipse = patches.Ellipse((lx, ly), width, height, angle=angle,
                                    fill=False, color='red', linestyle='--', 
                                    linewidth=1.5, alpha=0.7)
            self.ax.add_patch(ellipse)
            self.landmark_ellipses.append(ellipse)
    
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

            points, _ = scans.scan_to_cartesian(result['scan'], pose)
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
        points, _ = scans.scan_to_cartesian(result['scan'], pose)
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