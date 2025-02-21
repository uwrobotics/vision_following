#!/usr/bin/env python3

import sys
import math
import cv2
import numpy as np

def detect_tag_2d_angle(image_path):
    """
    1. Loads an image and detects the first 4x4 ArUco marker.
    2. Finds the marker center and computes an angle (in degrees) where:
         - 0° = marker is straight up from image center
         - +angle = marker is to the left of up
         - -angle = marker is to the right of up
    3. Draws a short arrow from the image center toward the marker
       and saves the result as '2d_angle_result.jpg'.
    """
    # --- Load the image ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'. Check the file path.")
        return

    # Image dimensions and center
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary (4x4 markers, 50 IDs)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Create ArUco detector parameters
    try:
        parameters = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        parameters = cv2.aruco.DetectorParameters()

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        # Draw detected markers on the image
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Use the first detected marker for demonstration
        marker_corners = corners[0].reshape((4, 2))
        # Marker center in the image
        mx, my = marker_corners.mean(axis=0)

        # -----------------------------
        # 1) Compute the image offsets
        # -----------------------------
        dx = mx - cx   # horizontal offset (+ right, - left in image coords)
        dy = my - cy   # vertical offset (+ down, - up in image coords)

        # ---------------------------------------------
        # 2) Convert to an "angle from straight up" 
        #
        #    We want:
        #      - 0° = marker is directly above center
        #      - +° = marker is left of up
        #      - -° = marker is right of up
        #
        #    In a standard math axis (x right, y up), we define:
        #      x' = dx, y' = -dy
        #    Then an angle from the y'-axis is given by: -atan2(x', y')
        # ---------------------------------------------
        x_prime = dx
        y_prime = -dy

        # Angle in radians, range (-π, π)
        # We use atan2(x', y') so that x'>0 => positive angle => 
        # but then we negate it to make x'>0 => negative angle => right.
        angle_radians = -math.atan2(x_prime, y_prime)
        angle_degrees = math.degrees(angle_radians)

        # Print the angle
        print(f"Marker angle from up: {angle_degrees:.2f}°")
        #  0°   => straight up
        #  +θ°  => left of up
        #  -θ°  => right of up

        # -----------------------------------------------------
        # 3) Draw a short arrow from center toward the marker
        #    so we can visually confirm the direction is correct.
        # -----------------------------------------------------
        arrow_length = 100.0  # in pixels
        dist_to_marker = math.sqrt(dx**2 + dy**2)
        if dist_to_marker > 1e-6:
            scale = arrow_length / dist_to_marker
        else:
            scale = 0.0  # marker is basically at center

        # End point of the arrow
        end_x = int(cx + scale * dx)
        end_y = int(cy + scale * dy)

        cv2.arrowedLine(
            image,
            (int(cx), int(cy)),
            (end_x, end_y),
            (0, 0, 255),
            2,
            tipLength=0.2
        )
    else:
        print("No ArUco markers detected.")

    # Save the result
    output_file = "2d_angle_result.jpg"
    cv2.imwrite(output_file, image)
    print(f"Annotated image saved as '{output_file}'")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_tag_2d_angle(image_path)

if __name__ == "__main__":
    main()
