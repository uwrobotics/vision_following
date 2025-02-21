#!/usr/bin/env python3
import sys
import math
import cv2
import numpy as np
import pyrealsense2 as rs

# Import GStreamer Python bindings
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# CHANGE THIS TO THE RECEIVER DEVICE'S IP ADDRESS ON YOUR LAN
TARGET_IP = "192.168.1.100"  
TARGET_PORT = 5000

def estimate_distance_3d(marker_corners, marker_size, camera_matrix, dist_coeffs):
    """
    Estimate the distance from the camera to the marker using solvePnP.
    
    marker_corners: (4,2) array of the marker's corner coordinates in the image.
    marker_size: Real-world side length of the marker (meters).
    camera_matrix: 3x3 camera calibration matrix.
    dist_coeffs: Distortion coefficients.
    
    Returns:
      distance (float): Euclidean distance (meters) from camera to marker,
                        or None if solvePnP fails.
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
    return float(np.linalg.norm(tvec))

def create_gst_pipeline(width, height, fps, target_ip, target_port):
    """
    Create and return a GStreamer pipeline for streaming.
    The pipeline uses appsrc to receive frames, encodes them with H.264,
    and sends them via UDP to the specified IP and port.
    """
    pipeline_str = (
        "appsrc name=source is-live=true block=true do-timestamp=true "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! queue ! "
        "x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
        "rtph264pay config-interval=1 ! "
        f"udpsink host={target_ip} port={target_port}"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("source")
    pipeline.set_state(Gst.State.PLAYING)
    return pipeline, appsrc

def main():
    # Initialize GStreamer
    Gst.init(None)

    # Set up RealSense pipeline and start streaming color frames
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_profile = rs_pipeline.start(rs_config)

    # Get RealSense camera intrinsics for the color stream
    color_profile = rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    camera_matrix = np.array([[fx,  0,  cx],
                              [0,   fy, cy],
                              [0,    0,  1]], dtype=np.float32)
    dist_coeffs = np.array([intr.coeffs[0],
                            intr.coeffs[1],
                            intr.coeffs[2],
                            intr.coeffs[3],
                            intr.coeffs[4]], dtype=np.float32)

    # Set up GStreamer pipeline for streaming over LAN
    width, height, fps = 640, 480, 30
    gst_pipeline, appsrc = create_gst_pipeline(width, height, fps, TARGET_IP, TARGET_PORT)

    # Prepare ArUco detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        aruco_params = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters()
    marker_size = 0.10  # Marker size in meters (adjust to your marker)

    while True:
        # Capture a frame from RealSense
        frames = rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Rotate the frame 180° to correct the physically inverted input
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # AR Tag Detection & Annotation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            marker_corners = corners[0]

            # Compute marker center in 2D
            m_center = marker_corners.reshape((4,2)).mean(axis=0)
            mx, my = m_center
            cx_img, cy_img = width / 2, height / 2
            dx = mx - cx_img
            dy = my - cy_img

            # Compute 2D angle from "straight up" (0° means directly above center)
            # Use x' = dx and y' = -dy (flip the y axis)
            x_prime = dx
            y_prime = -dy
            angle_radians = -math.atan2(x_prime, y_prime)
            angle_degrees = math.degrees(angle_radians)

            # Compute 3D distance using solvePnP
            distance = estimate_distance_3d(marker_corners, marker_size, camera_matrix, dist_coeffs)

            # Draw an arrow from image center to marker center (scaled to 100 pixels)
            arrow_length = 100
            dist_2d = math.sqrt(dx*dx + dy*dy)
            scale = arrow_length / dist_2d if dist_2d > 1e-6 else 0
            end_pt = (int(cx_img + scale*dx), int(cy_img + scale*dy))
            cv2.arrowedLine(frame, (int(cx_img), int(cy_img)), end_pt, (0, 0, 255), 2, tipLength=0.2)

            # Overlay text with distance and angle
            if distance is not None:
                text_str = f"{distance:.2f}m, {angle_degrees:.2f}°"
            else:
                text_str = f"??m, {angle_degrees:.2f}°"
            cv2.putText(frame, text_str, (int(cx_img)-80, int(cy_img)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Send the annotated frame into the GStreamer pipeline
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        timestamp = Gst.util_get_timestamp()
        buf.pts = timestamp
        buf.dts = timestamp
        buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        appsrc.emit("push-buffer", buf)

if __name__ == "__main__":
    main()
