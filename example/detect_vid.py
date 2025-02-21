#!/usr/bin/env python3

import sys
import math
import cv2
import numpy as np

def estimate_distance_3d(marker_corners, marker_size, camera_matrix, dist_coeffs):
    """
    Use solvePnP to estimate the distance from the camera to the marker.
    
    marker_corners: (4,2) array of the marker's corner coordinates in the image.
    marker_size: The real-world size of the marker's side (in meters).
    camera_matrix: 3×3 camera calibration matrix.
    dist_coeffs: Distortion coefficients.
    
    Returns:
      distance (float): Euclidean distance from camera to marker (in meters).
                       None if solvePnP fails.
    """
    object_points = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    image_points = marker_corners.reshape((4, 2)).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None

    distance = np.linalg.norm(tvec)
    return distance

def process_video(input_video_path, output_video_path):
    """
    Reads an .mp4 video, detects a 4×4 ArUco marker in each frame,
    and computes both:
      - 2D angle from "straight up" in the image (left/right)
      - 3D distance using solvePnP
    Overlays an arrow + text onto each frame, then inverts (rotates 180°)
    the annotated frame before writing it to output_video_path.
    """

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_video_path}'.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        parameters = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        parameters = cv2.aruco.DetectorParameters()

    # Known or estimated camera parameters (tweak for your setup)
    marker_size = 0.10  # e.g., 10 cm
    fx = 600.0
    fy = 600.0
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([
        [fx,   0.0, cx ],
        [0.0,  fy,  cy ],
        [0.0,  0.0, 1.0]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Use the first detected marker for demonstration
            marker_corners = corners[0]
            # --- 2D Angle ---
            mx, my = marker_corners.reshape((4,2)).mean(axis=0)
            cx_img, cy_img = width / 2, height / 2
            dx = mx - cx_img
            dy = my - cy_img

            # "Angle from up": 0° => up, +° => left, -° => right
            x_prime = dx
            y_prime = -dy
            angle_radians = -math.atan2(x_prime, y_prime)
            angle_degrees = math.degrees(angle_radians)

            # --- 3D Distance ---
            distance = estimate_distance_3d(marker_corners, marker_size, camera_matrix, dist_coeffs)

            # Draw arrow from center to marker
            arrow_length = 100
            dist_2d = math.sqrt(dx*dx + dy*dy)
            scale = arrow_length / dist_2d if dist_2d > 1e-6 else 0
            end_x = int(cx_img + scale * dx)
            end_y = int(cy_img + scale * dy)
            cv2.arrowedLine(frame, (int(cx_img), int(cy_img)), (end_x, end_y), (0, 0, 255), 2, tipLength=0.2)

            # Overlay text
            if distance is not None:
                text_str = f"{distance:.2f}m, {angle_degrees:.2f}°"
            else:
                text_str = f"??m, {angle_degrees:.2f}°"

            cv2.putText(
                frame,
                text_str,
                (int(cx_img) - 80, int(cy_img) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

        # Write the inverted annotated frame
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved inverted annotated video to '{output_video_path}'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_2d_angle_distance_video_inverted.py <input_video.mp4>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_video_path = "2d_angle_distance_result_inverted.mp4"
    process_video(input_video_path, output_video_path)

if __name__ == "__main__":
    main()
