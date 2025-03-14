#!/usr/bin/env python3
import time
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from abc import ABC, abstractmethod

DEBUG = True  # If True, we print debug logs

# ---------------------------------------
# CONFIG
# ---------------------------------------
# Camera field-of-view parameters
DEFAULT_HFOV = 60.7      # Horizontal FOV in degrees (wide)
DEFAULT_VFOV = 34.0      # Vertical FOV in degrees (approximate for 16:9 sensor)

# Partial panorama coverage (based on physical PT range)
PAN_MIN, PAN_MAX = -135.0, 135.0   # 270° total
TILT_MIN, TILT_MAX = -30.0, 90.0     # 120° total

# Output canvas resolution for partial panorama (flat equirectangular)
OUT_WIDTH, OUT_HEIGHT = 2700, 1200

# ---------------------------------------
# FreeD Packet Parsing (Unchanged)
# ---------------------------------------
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
# Tile Stamping Approach (Flat Partial Panorama)
# ---------------------------
def tile_stamp(
    img: np.ndarray,
    center_pan: float,
    center_tilt: float,
    hfov: float,
    vfov: float,
    pano_map: np.ndarray,
    pano_mask: np.ndarray,
    pan_min=PAN_MIN,
    pan_max=PAN_MAX,
    tilt_min=TILT_MIN,
    tilt_max=TILT_MAX
):
    """
    Stamp `img` onto the partial equirectangular canvas.
    The snapshot covers:
       [center_pan - hfov/2, center_pan + hfov/2] in pan, and
       [center_tilt - vfov/2, center_tilt + vfov/2] in tilt.
    Overlapping areas are alpha-blended 50/50.
    Vertical mapping: tilt = TILT_MAX (+90°) maps to y = 0 (top),
                      tilt = TILT_MIN (-30°) maps to y = pano_h (bottom).
    """
    h, w = img.shape[:2]
    pano_h, pano_w = pano_map.shape[:2]

    # Compute bounding box in (lon, lat)
    lon0 = center_pan - hfov/2
    lon1 = center_pan + hfov/2
    lat0 = center_tilt - vfov/2
    lat1 = center_tilt + vfov/2

    if DEBUG:
        print(f"Tile bounding box: lon=[{lon0}, {lon1}], lat=[{lat0}, {lat1}]")

    for y_img in range(h):
        lat_deg = lat0 + (y_img / (h - 1)) * (lat1 - lat0)
        if lat_deg < tilt_min or lat_deg > tilt_max:
            continue
        # Map vertical so that tilt_max maps to y=0, tilt_min to y=pano_h
        y_e = ((tilt_max - lat_deg) / (tilt_max - tilt_min)) * pano_h
        yi = int(np.floor(y_e))
        if yi < 0 or yi >= pano_h:
            continue

        for x_img in range(w):
            lon_deg = lon0 + (x_img / (w - 1)) * (lon1 - lon0)
            if lon_deg < pan_min or lon_deg > pan_max:
                continue
            x_e = ((lon_deg - pan_min) / (pan_max - pan_min)) * pano_w
            xi = int(np.floor(x_e))
            if xi < 0 or xi >= pano_w:
                continue

            src_pixel = img[y_img, x_img]
            if pano_mask[yi, xi] == 0:
                pano_map[yi, xi] = src_pixel
                pano_mask[yi, xi] = 255
            else:
                existing = pano_map[yi, xi].astype(np.float32)
                pano_map[yi, xi] = (existing + src_pixel.astype(np.float32)) / 2

# ---------------------------
# Partial Stitching with Tile Approach
# ---------------------------
def partial_stitch_images(
    image_list,
    freeD_data_list,
    out_width=OUT_WIDTH,
    out_height=OUT_HEIGHT,
    default_hfov=DEFAULT_HFOV,
    default_vfov=DEFAULT_VFOV
):
    """
    Build a partial equirectangular panorama covering:
       Pan: -135..+135, Tilt: -30..+90.
    Each snapshot is placed as a tile based on its center pan/tilt.
    Overlapping areas are alpha-blended.
    """
    pano_map = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    pano_mask = np.zeros((out_height, out_width), dtype=np.uint8)
    total = len(image_list)
    for idx, (img, meta) in enumerate(zip(image_list, freeD_data_list)):
        pan_center = meta.get('pan', 0.0)
        tilt_center = meta.get('tilt', 0.0)
        # Flip vertically to correct each tile's orientation
        flipped_img = cv2.flip(img, 0)
        tile_stamp(
            flipped_img,
            center_pan=pan_center,
            center_tilt=tilt_center,
            hfov=default_hfov,
            vfov=default_vfov,
            pano_map=pano_map,
            pano_mask=pano_mask
        )
        progress = int((idx + 1) / total * 100)
        print(f"Stitching snapshot {idx+1}/{total} ({progress}% complete)")
        cv2.imshow("Live Partial Pano", pano_map)
        cv2.waitKey(1)
    return pano_map

# ---------------------------
# Spherical Warp Function
# ---------------------------
def warp_partial_to_spherical(pano, output_size=(600, 600)):
    """
    Warp the flat partial panorama (pano) into a spherical projection.
    We assume the input pano covers:
      - Longitude: PAN_MIN to PAN_MAX (i.e. -135° to +135°)
      - Latitude: TILT_MIN to TILT_MAX (i.e. -30° to +90°)
    
    The output is created using an inverse mapping.
    For each pixel (x, y) in the output, we compute normalized coordinates
    and then derive spherical angles using a stereographic-like projection.
    
    This is a basic implementation and may require tuning.
    """
    out_w, out_h = output_size
    spherical = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    # Create mapping arrays
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    # Define the output image center
    cx_out, cy_out = out_w / 2.0, out_h / 2.0
    # For each output pixel, compute normalized coordinates in [-1, 1]
    for y in range(out_h):
        for x in range(out_w):
            # Normalize coordinates so that center = 0
            u = (x - cx_out) / cx_out
            v = (y - cy_out) / cy_out
            r = np.sqrt(u*u + v*v)
            if r > 1:
                # Outside of the unit circle: leave black
                map_x[y, x] = -1
                map_y[y, x] = -1
            else:
                # Compute spherical coordinates via inverse stereographic projection:
                # theta: angle from the north pole; phi: azimuth (longitude)
                theta = 2 * np.arctan(r)
                phi = np.arctan2(v, u)
                # In spherical coordinates, latitude = 90° - theta (in degrees)
                lat = 90 - np.degrees(theta)
                # And longitude = phi in degrees.
                lon = np.degrees(phi)
                # Now, remap these spherical angles into the partial pano coordinate system.
                # Our input pano covers:
                #   lon: PAN_MIN .. PAN_MAX, lat: TILT_MIN .. TILT_MAX.
                if lon < PAN_MIN or lon > PAN_MAX or lat < TILT_MIN or lat > TILT_MAX:
                    map_x[y, x] = -1
                    map_y[y, x] = -1
                else:
                    # Map lon to x coordinate in pano:
                    x_in = (lon - PAN_MIN) / (PAN_MAX - PAN_MIN) * pano.shape[1]
                    # For latitude, since our pano y=0 corresponds to TILT_MAX,
                    # and y = pano_h corresponds to TILT_MIN, we map accordingly.
                    y_in = ((TILT_MAX - lat) / (TILT_MAX - TILT_MIN)) * pano.shape[0]
                    map_x[y, x] = x_in
                    map_y[y, x] = y_in

    # Use remap to create the spherical projection
    spherical = cv2.remap(pano, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return spherical

# ---------------------------
# Helper Functions for Absolute Positioning
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
# Camera Control & Snapshot Capture
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
# Main Capture Routine
# ---------------------------
def main_capture():
    """
    Capture snapshots over pan=-135..+135 and tilt=-30..+90, stitch them into a
    partial equirectangular (270°x120°) panorama using a tile-based approach,
    then warp the flat panorama into a spherical projection.
    """
    pan_start, pan_end = -135, 135
    tilt_start, tilt_end = -30, 90
    h_step = 45.0  # 45° steps horizontally
    v_step = 30.0  # 30° steps vertically

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
        time.sleep(2)
        img, meta = capture_snapshot()
        if img is not None:
            meta.update(pos)
            captured_images.append(img)
            metadata_list.append(meta)
            print(f"Snapshot captured at {pos}")
        else:
            print(f"Snapshot capture failed at {pos}")

    for i, im in enumerate(captured_images):
        cv2.imwrite(f"snapshot_{i}.jpg", im)

    # Build the flat partial panorama
    flat_pano = partial_stitch_images(
        captured_images,
        metadata_list,
        out_width=OUT_WIDTH,
        out_height=OUT_HEIGHT,
        default_hfov=DEFAULT_HFOV,
        default_vfov=DEFAULT_VFOV
    )
    cv2.imwrite("partial_panorama.jpg", flat_pano)
    cv2.imshow("Flat Partial Panorama", flat_pano)
    cv2.waitKey(1000)

    # Warp the flat panorama into a spherical projection
    spherical_pano = warp_partial_to_spherical(flat_pano, output_size=(600,600))
    cv2.imwrite("spherical_panorama.jpg", spherical_pano)
    cv2.imshow("Spherical Panorama", spherical_pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_capture()
