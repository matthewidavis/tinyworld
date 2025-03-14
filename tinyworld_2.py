#!/usr/bin/env python3
import time
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from abc import ABC, abstractmethod

DEBUG = True  # Set True to draw corners & debug prints

# ---------------------------------------
# CONFIG: approximate lens offset in inches
# ---------------------------------------
LENS_OFFSET_IN = 1.5      # 1.5 inches from pivot to lens
PX_PER_INCH    = 100.0    # You MUST adjust this scale factor as needed!
                         # e.g. if 1 inch ~ 100 "pixel-units"

# ---------------------------
# FreeD Packet Parsing (Unchanged)
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
    return combined_fixed / 32768.0

def pose(byte_0: int, byte_1: int, byte_2: int) -> float:
    fixed = (byte_0 << 16) | (byte_1 << 8) | byte_2
    if fixed & 0x800000:
        fixed -= 0x1000000
    integer_part = fixed >> 6
    fractional_part = fixed & 0x3F
    combined_fixed = (integer_part << 6) | fractional_part
    return combined_fixed / 64.0

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
# Partial Equirectangular Stitching
# ---------------------------
def project_image_to_partial_sphere(
    img: np.ndarray,
    pan_deg: float,
    tilt_deg: float,
    hfov_deg: float,
    pano_map: np.ndarray,
    pano_mask: np.ndarray,
    lens_offset_inch: float,
    px_per_inch: float,
    overwrite: bool = False
):
    """
    Project a snapshot onto a PARTIAL equirectangular canvas:
      - Horizontal (pan) range: -135..+135
      - Vertical (tilt) range: -30..+90
    This is 270° horizontally and 120° vertically.

    We incorporate a lens offset from the pivot (in inches) to better model real PTZ geometry.
    lens_offset_inch: how many inches forward the lens is from the pivot
    px_per_inch: scale factor to convert inches -> "pixel-units" (approx).
    """
    # Pan/Tilt range for the final partial sphere
    PAN_MIN, PAN_MAX = -135.0, 135.0   # 270° total
    TILT_MIN, TILT_MAX = -30.0, 90.0   # 120° total

    h, w = img.shape[:2]
    pano_h, pano_w = pano_map.shape[:2]

    # Convert angles to radians
    pan_rad = np.deg2rad(pan_deg)
    # Negative tilt => "tilt up" is positive lat
    tilt_rad = -np.deg2rad(tilt_deg)
    hfov_rad = np.deg2rad(hfov_deg)
    focal = w / (2.0 * np.tan(hfov_rad / 2.0))

    cx, cy = w / 2.0, h / 2.0

    # Convert lens offset (inches) to pixel-based coords
    # We'll assume the lens offset is purely along +Z (forward),
    # but you might need [x, y, z] offsets if the pivot is above or below the lens, etc.
    lens_offset_px = np.array([0.0, 0.0, lens_offset_inch * px_per_inch], dtype=np.float32)

    # Build rotation matrix
    Rz = np.array([
        [np.cos(pan_rad), -np.sin(pan_rad), 0],
        [np.sin(pan_rad),  np.cos(pan_rad), 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad),  np.cos(tilt_rad)]
    ])
    R = Rz @ Rx

    # Debug: compute projected corners
    projected_corners = []
    if DEBUG:
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        for (x_img, y_img) in corners:
            dx = x_img - cx
            dy = y_img - cy
            # lens-ray in pixel coords
            ray_lens = np.array([dx, -dy, focal], dtype=np.float32)
            # shift by lens offset
            ray_pivot = lens_offset_px + ray_lens
            # rotate about pivot
            ray_global = R @ ray_pivot
            norm = np.linalg.norm(ray_global)
            if norm < 1e-6:
                projected_corners.append((-999, -999))
                continue
            ray_global /= norm
            lon_deg = np.rad2deg(np.arctan2(ray_global[1], ray_global[0]))
            lat_deg = np.rad2deg(np.arcsin(ray_global[2]))
            if not (PAN_MIN <= lon_deg <= PAN_MAX and TILT_MIN <= lat_deg <= TILT_MAX):
                projected_corners.append((-999, -999))
                continue
            # map to partial canvas
            x_e = (lon_deg - PAN_MIN) / (PAN_MAX - PAN_MIN) * pano_w
            y_e = ((lat_deg - TILT_MIN) / (TILT_MAX - TILT_MIN)) * pano_h
            projected_corners.append((int(x_e), int(y_e)))
        print(f"Image corners (partial) w/offset at pan {pan_deg}, tilt {tilt_deg}: {projected_corners}")

    # For each pixel in the snapshot
    for y_img in range(h):
        for x_img in range(w):
            dx = x_img - cx
            dy = y_img - cy
            ray_lens = np.array([dx, -dy, focal], dtype=np.float32)
            ray_pivot = lens_offset_px + ray_lens
            ray_global = R @ ray_pivot
            norm = np.linalg.norm(ray_global)
            if norm < 1e-6:
                continue
            ray_global /= norm

            lon_deg = np.rad2deg(np.arctan2(ray_global[1], ray_global[0]))
            lat_deg = np.rad2deg(np.arcsin(ray_global[2]))

            # Skip if outside partial coverage
            if not (PAN_MIN <= lon_deg <= PAN_MAX):
                continue
            if not (TILT_MIN <= lat_deg <= TILT_MAX):
                continue

            x_e = (lon_deg - PAN_MIN) / (PAN_MAX - PAN_MIN) * pano_w
            y_e = ((TILT_MAX - lat_deg) / (TILT_MAX - TILT_MIN)) * pano_h
            xi = int(np.floor(x_e))
            yi = int(np.floor(y_e))

            if 0 <= xi < pano_w and 0 <= yi < pano_h:
                src_pixel = img[y_img, x_img]
                if overwrite:
                    pano_map[yi, xi] = src_pixel
                    pano_mask[yi, xi] = 255
                else:
                    if pano_mask[yi, xi] == 0:
                        pano_map[yi, xi] = src_pixel
                        pano_mask[yi, xi] = 255
                    else:
                        existing = pano_map[yi, xi].astype(np.float32)
                        pano_map[yi, xi] = (existing + src_pixel.astype(np.float32)) / 2

    if DEBUG:
        for (xi, yi) in projected_corners:
            if xi >= 0 and yi >= 0:
                cv2.circle(pano_map, (xi, yi), 5, (0, 0, 255), -1)


def partial_stitch_images(
    image_list,
    freeD_data_list,
    out_width=2700,
    out_height=1200,
    default_hfov=60.7,
    lens_offset_in=LENS_OFFSET_IN,
    px_per_inch=PX_PER_INCH
):
    """
    Build a partial equirectangular panorama covering:
      - Pan: -135..+135
      - Tilt: -30..+90
    = 270° horizontally x 120° vertically

    lens_offset_in: how many inches forward the lens is from the pivot
    px_per_inch: scale factor to convert inches to pixel-based coords
    """
    pano_map = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    pano_mask = np.zeros((out_height, out_width), dtype=np.uint8)

    total = len(image_list)
    for idx, (img, meta) in enumerate(zip(image_list, freeD_data_list)):
        pan = meta.get('pan', 0.0)
        tilt = meta.get('tilt', 0.0)
        project_image_to_partial_sphere(
            img, pan, tilt, default_hfov,
            pano_map, pano_mask,
            lens_offset_inch=lens_offset_in,
            px_per_inch=px_per_inch,
            overwrite=False
        )
        progress = int((idx + 1) / total * 100)
        print(f"Stitching snapshot {idx+1}/{total} ({progress}% complete)")

        # Live update
        cv2.imshow("Live Partial Pano", pano_map)
        cv2.waitKey(1)

    return pano_map

# ---------------------------
# Helper Functions for Absolute Positioning
# ---------------------------
def convert_pan_angle(angle: float) -> str:
    """
    Convert a pan angle in degrees to 4-digit hex for the camera's absolute command.
    2's complement is handled by subtracting from 65535 if negative.
    """
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
    """
    Convert a tilt angle in degrees to 4-digit hex for the camera's absolute command.
    2's complement is handled by subtracting from 65535 if negative.
    """
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
# Camera Control & Snapshot
# ---------------------------
camera_ip = "192.168.12.8"  # your camera's IP
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
        arr = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        metadata = {}
        return img, metadata
    else:
        return None, None

# ---------------------------
# Main Capture
# ---------------------------
def main_capture():
    """
    Capture snapshots over pan=±135, tilt=-30..+90, and stitch them
    into a partial equirectangular (270x120) panorama, accounting for
    a 1.5" lens offset from the pivot (approx).
    """
    # define grid
    pan_start, pan_end = -135, 135
    tilt_start, tilt_end = -30, 90
    h_step = 45.0  # e.g., 45° steps horizontally
    v_step = 30.0  # e.g., 30° steps vertically

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
        print(f"Moving camera to: {pos}")
        if not move_camera_absolute(pos['pan'], pos['tilt']):
            print(f"Failed to move camera: {pos}")
            continue
        time.sleep(2)  # wait for PTZ to settle
        img, meta = capture_snapshot()
        if img is not None:
            meta.update(pos)
            captured_images.append(img)
            metadata_list.append(meta)
            print(f"Snapshot captured at {pos}")
        else:
            print(f"Snapshot capture failed at {pos}")

    # optional: save snapshots for debug
    for i, im in enumerate(captured_images):
        cv2.imwrite(f"snapshot_{i}.jpg", im)

    # partial equirect stitching with lens offset
    out_w, out_h = 2700, 1200
    pano = partial_stitch_images(
        captured_images,
        metadata_list,
        out_width=out_w,
        out_height=out_h,
        default_hfov=60.7,
        lens_offset_in=LENS_OFFSET_IN,
        px_per_inch=PX_PER_INCH
    )

    cv2.imwrite("partial_panorama.jpg", pano)
    cv2.imshow("Partial Panorama", pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_capture()
