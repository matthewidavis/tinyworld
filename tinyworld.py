#!/usr/bin/env python3
import time
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from abc import ABC, abstractmethod

DEBUG = True  # Set to True to draw projected corners on the sphere map

# ---------------------------
# FreeD Packet Parsing Section (Unchanged)
# ---------------------------
class FreeDPacket:
    def __init__(self, msg_type: int, cam_id: int,
                 pan: float, tilt: float, roll: float,
                 pos_x: float, pos_y: float, pos_z: float,
                 zoom: int, focus: int,
                 checksum: int):
        self.msg_type = msg_type
        self.cam_id = cam_id
        self.pan = pan
        self.tilt = tilt
        self.roll = roll
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.zoom = zoom
        self.focus = focus
        self.checksum = checksum

def angle(byte_0: int, byte_1: int, byte_2: int) -> float:
    fixed = (byte_0 << 16) | (byte_1 << 8) | byte_2
    if fixed & 0x800000:
        fixed -= 0x1000000
    integer_part = fixed >> 15
    fractional_part = fixed & 0x7FFF
    combined_fixed = (integer_part << 15) | fractional_part
    float_value = combined_fixed / 32768.0
    return float_value

def pose(byte_0: int, byte_1: int, byte_2: int) -> float:
    fixed = (byte_0 << 16) | (byte_1 << 8) | byte_2
    if fixed & 0x800000:
        fixed -= 0x1000000
    integer_part = fixed >> 6
    fractional_part = fixed & 0x3F
    combined_fixed = (integer_part << 6) | fractional_part
    float_value = combined_fixed / 64.0
    return float_value

def int_from_3bytes(byte_0: int, byte_1: int, byte_2: int) -> int:
    return (byte_0 << 16) | (byte_1 << 8) | byte_2

def freed_packet_from_bytes(byte_list: list[int]) -> FreeDPacket:
    message_type = byte_list[0]
    cam_id = byte_list[1]
    pan = angle(byte_list[2], byte_list[3], byte_list[4])
    tilt = angle(byte_list[5], byte_list[6], byte_list[7])
    roll = angle(byte_list[8], byte_list[9], byte_list[10])
    x = pose(byte_list[11], byte_list[12], byte_list[13])
    y = pose(byte_list[14], byte_list[15], byte_list[16])
    z = pose(byte_list[17], byte_list[18], byte_list[19])
    zoom = int_from_3bytes(byte_list[20], byte_list[21], byte_list[22])
    focus = int_from_3bytes(byte_list[24], byte_list[25], byte_list[26])
    checksum = byte_list[28]
    return FreeDPacket(message_type, cam_id, pan, tilt, roll, x, y, z, zoom, focus, checksum)

class FreeD(ABC):
    @abstractmethod
    def start(self) -> None:
        pass
    @abstractmethod
    def stop(self) -> None:
        pass
    @abstractmethod
    def receive_packet(self) -> FreeDPacket:
        pass

# ---------------------------
# Spherical Stitching Section with Debug Visualization
# ---------------------------
def project_image_to_sphere(
    img: np.ndarray,
    pan_deg: float,
    tilt_deg: float,
    hfov_deg: float,
    sphere_map: np.ndarray,
    sphere_mask: np.ndarray,
    overwrite: bool = False
):
    """
    Project a snapshot onto an equirectangular sphere_map.
    - img: source snapshot (BGR)
    - pan_deg, tilt_deg: camera orientation (in degrees)
    - hfov_deg: horizontal field of view for this snapshot (in degrees)
    - sphere_map: equirectangular canvas (H x W x 3)
    - sphere_mask: mask array (H x W) tracking filled pixels
    - overwrite: if True, overwrite existing pixels; else do simple averaging.
    
    The equirectangular coordinates:
      - x in [0, width) maps to longitude in [-π, π)
      - y in [0, height) maps to latitude in [π/2, -π/2]
    """
    h, w = img.shape[:2]
    # Convert angles to radians.
    pan_rad = np.deg2rad(pan_deg)
    # Invert tilt so that "tilt up" increases latitude.
    tilt_rad = -np.deg2rad(tilt_deg)
    hfov_rad = np.deg2rad(hfov_deg)
    # Approximate focal length using the pinhole model.
    focal = w / (2.0 * np.tan(hfov_rad / 2.0))
    sphere_h, sphere_w = sphere_map.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Build rotation matrices.
    Rz = np.array([
        [ np.cos(pan_rad), -np.sin(pan_rad), 0],
        [ np.sin(pan_rad),  np.cos(pan_rad), 0],
        [ 0,                0,               1]
    ])
    Rx = np.array([
        [1,               0,                0],
        [0,  np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0,  np.sin(tilt_rad),  np.cos(tilt_rad)]
    ])
    # Use rotation: first tilt then pan.
    R = Rz @ Rx

    # Debug: Compute projected corners for this image.
    if DEBUG:
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        projected_corners = []
        for (x_img, y_img) in corners:
            dx = x_img - cx
            dy = y_img - cy
            ray_cam = np.array([dx, -dy, focal])
            ray_global = R @ ray_cam
            norm = np.linalg.norm(ray_global)
            if norm < 1e-6:
                continue
            ray_global /= norm
            Xg, Yg, Zg = ray_global
            lon = np.arctan2(Yg, Xg)
            lat = np.arcsin(Zg)
            x_e = (lon + np.pi) / (2.0 * np.pi) * sphere_w
            y_e = ((np.pi/2.0) - lat) / np.pi * sphere_h
            projected_corners.append((int(x_e), int(y_e)))
        print(f"Image corners projected at pan {pan_deg}°, tilt {tilt_deg}°: {projected_corners}")

    # Iterate over each pixel in the source image.
    for y_img in range(h):
        for x_img in range(w):
            dx = x_img - cx
            dy = y_img - cy
            ray_cam = np.array([dx, -dy, focal])
            ray_global = R @ ray_cam
            norm = np.linalg.norm(ray_global)
            if norm < 1e-6:
                continue
            ray_global /= norm
            Xg, Yg, Zg = ray_global
            lon = np.arctan2(Yg, Xg)
            lat = np.arcsin(Zg)
            x_e = (lon + np.pi) / (2.0 * np.pi) * sphere_w
            y_e = ((np.pi/2.0) - lat) / np.pi * sphere_h
            xi = int(np.floor(x_e))
            yi = int(np.floor(y_e))
            if 0 <= xi < sphere_w and 0 <= yi < sphere_h:
                src_pixel = img[y_img, x_img, :]
                if overwrite:
                    sphere_map[yi, xi, :] = src_pixel
                    sphere_mask[yi, xi] = 255
                else:
                    if sphere_mask[yi, xi] == 0:
                        sphere_map[yi, xi, :] = src_pixel
                        sphere_mask[yi, xi] = 255
                    else:
                        existing = sphere_map[yi, xi, :].astype(np.float32)
                        sphere_map[yi, xi, :] = (existing + src_pixel.astype(np.float32)) / 2

    # If debugging, draw the projected corner points onto the sphere map.
    if DEBUG:
        for (xi, yi) in projected_corners:
            cv2.circle(sphere_map, (xi, yi), 5, (0, 0, 255), -1)  # red circles

def spherical_stitch_images(
    image_list,
    freeD_data_list,
    out_width=4096,
    out_height=2048,
    default_hfov=60.7
):
    """
    Create an equirectangular panorama from a set of images and their pan/tilt metadata.
    - out_width, out_height: dimensions of the output panorama.
    - default_hfov: horizontal FOV (in degrees) for the wide (unzoomed) setting.
    
    Prints a progress update for each snapshot processed.
    """
    sphere_map = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    sphere_mask = np.zeros((out_height, out_width), dtype=np.uint8)
    total = len(image_list)
    
    for idx, (img, meta) in enumerate(zip(image_list, freeD_data_list)):
        pan = meta.get('pan', 0.0)
        tilt = meta.get('tilt', 0.0)
        hfov = default_hfov  # Use the wide-field HFOV
        project_image_to_sphere(img, pan, tilt, hfov, sphere_map, sphere_mask, overwrite=False)
        progress = int((idx + 1) / total * 100)
        print(f"Stitching snapshot {idx+1}/{total} ({progress}% complete)")
    
    return sphere_map

# ---------------------------
# Helper Functions for Absolute Positioning (Unchanged)
# ---------------------------
def convert_pan_angle(angle: float) -> str:
    factor = 14.4
    if angle >= 0:
        dec_value = int(angle * factor)
        dec_value = min(dec_value, 2448)
        hex_str = format(dec_value, '04X')
    else:
        dec_value = 65535 - int(abs(angle) * factor)
        hex_str = format(dec_value, '04X')
    return hex_str

def convert_tilt_angle(angle: float) -> str:
    factor = 14.4
    if angle >= 0:
        dec_value = int(angle * factor)
        dec_value = min(dec_value, 1296)
        hex_str = format(dec_value, '04X')
    else:
        dec_value = 65535 - int(abs(angle) * factor)
        hex_str = format(dec_value, '04X')
    return hex_str

# ---------------------------
# Camera Control and Snapshot Capture Section (Unchanged)
# ---------------------------
camera_ip = "192.168.12.8"
username = "admin"
password = "2Password2!"

def move_camera_absolute(pan_angle: float, tilt_angle: float) -> bool:
    pan_hex = convert_pan_angle(pan_angle)
    tilt_hex = convert_tilt_angle(tilt_angle)
    url = (f"http://{camera_ip}/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&"
           f"{pan_hex}&{tilt_hex}")
    print(f"Sending command: {url}")
    response = requests.get(url, auth=HTTPDigestAuth(username, password))
    return response.ok

def capture_snapshot():
    snapshot_url = f"http://{camera_ip}/snapshot.jpg"
    response = requests.get(snapshot_url, auth=HTTPDigestAuth(username, password))
    if response.ok:
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        metadata = {}
        return image, metadata
    else:
        return None, None

# ---------------------------
# Main Capture Routine (Using Spherical Stitching with Debug)
# ---------------------------
def main_capture():
    """
    Move the camera over a grid covering 340° pan and 120° tilt,
    capture snapshots, and stitch them into an equirectangular panorama.
    
    Pan range: -170° to +170°
    Tilt range: -30° (down) to +90° (up)
    
    Using a ~30% overlap:
      - Horizontal step ≈ HFOV * 0.7 ≈ 60.7 * 0.7 ≈ 42.5°
      - Vertical step ≈ VFOV * 0.7 ≈ 34.1 * 0.7 ≈ 23.9°
    """
    # Define grid based on full motion range.
    pan_start, pan_end = -170, 170
    tilt_start, tilt_end = -30, 90
    h_step = 42.5  # horizontal step (degrees)
    v_step = 23.9  # vertical step (degrees)
    
    pan_values = np.arange(pan_start, pan_end + 1, h_step)
    tilt_values = np.arange(tilt_start, tilt_end + 1, v_step)
    
    positions = []
    for tilt in tilt_values:
        for pan in pan_values:
            positions.append({'pan': float(pan), 'tilt': float(tilt)})
    
    print(f"Total positions: {len(positions)}")
    
    captured_images = []
    metadata_list = []
    
    for pos in positions:
        print(f"Moving camera to: Pan {pos['pan']}°, Tilt {pos['tilt']}°")
        if not move_camera_absolute(pos['pan'], pos['tilt']):
            print(f"Failed to move camera to position: {pos}")
            continue
        
        time.sleep(2)  # allow the camera to settle
        
        image, metadata = capture_snapshot()
        if image is not None:
            metadata.update(pos)
            captured_images.append(image)
            metadata_list.append(metadata)
            print(f"Snapshot captured at position: {pos}")
        else:
            print(f"Snapshot capture failed at position: {pos}")
    
    # Save individual snapshots for debugging.
    for i, img in enumerate(captured_images):
        cv2.imwrite(f"snapshot_{i}.jpg", img)
    
    # Spherical stitching with live progress and debug overlays.
    out_width = 4096
    out_height = 2048
    default_hfov = 60.7  # wide setting HFOV in degrees
    
    sphere_result = spherical_stitch_images(
        captured_images,
        metadata_list,
        out_width=out_width,
        out_height=out_height,
        default_hfov=default_hfov
    )
    
    cv2.imwrite("panorama_spherical_debug.jpg", sphere_result)
    cv2.imshow("Spherical Panorama (Debug)", sphere_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_capture()
