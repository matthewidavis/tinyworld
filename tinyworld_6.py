#!/usr/bin/env python3
import time
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from abc import ABC, abstractmethod

DEBUG = True

# ---------------------------
# CONFIG
# ---------------------------
DEFAULT_HFOV = 60.7
DEFAULT_VFOV = 34.0

# Partial coverage
PAN_MIN, PAN_MAX = -135.0, 135.0   # 270° total
TILT_MIN, TILT_MAX = -30.0, 90.0   # 120° total
OUT_WIDTH, OUT_HEIGHT = 2700, 1200  # partial equirect dimension

# Full 360° coverage for equirect (2:1)
FULL_W, FULL_H = 4000, 2000  # or 4096x2048, etc.

# ---------------------------
# Freed Packet Parsing
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
# Tile-based partial coverage
# ---------------------------
def tile_stamp(
    img: np.ndarray,
    center_pan: float,
    center_tilt: float,
    hfov: float,
    vfov: float,
    pano_map: np.ndarray,
    pano_mask: np.ndarray
):
    """
    Stamp each snapshot in partial equirect domain:
    pan in [-135..+135], tilt in [-30..+90].
    
    ***NO ALPHA BLENDING*** (last snapshot wins).
    """
    h, w = img.shape[:2]
    pano_h, pano_w = pano_map.shape[:2]

    lon0 = center_pan - hfov/2
    lon1 = center_pan + hfov/2
    lat0 = center_tilt - vfov/2
    lat1 = center_tilt + vfov/2

    for y_img in range(h):
        lat_deg = lat0 + (y_img / (h - 1)) * (lat1 - lat0)
        if lat_deg < TILT_MIN or lat_deg > TILT_MAX:
            continue
        y_e = ((TILT_MAX - lat_deg) / (TILT_MAX - TILT_MIN)) * pano_h
        yi = int(np.floor(y_e))
        if yi < 0 or yi >= pano_h:
            continue

        for x_img in range(w):
            lon_deg = lon0 + (x_img / (w - 1)) * (lon1 - lon0)
            if lon_deg < PAN_MIN or lon_deg > PAN_MAX:
                continue
            x_e = ((lon_deg - PAN_MIN) / (PAN_MAX - PAN_MIN)) * pano_w
            xi = int(np.floor(x_e))
            if xi < 0 or xi >= pano_w:
                continue

            src_pixel = img[y_img, x_img]
            # *** NO ALPHA BLEND ***
            pano_map[yi, xi] = src_pixel
            pano_mask[yi, xi] = 255

def partial_stitch_images(
    image_list,
    freeD_data_list,
    out_width=OUT_WIDTH,
    out_height=OUT_HEIGHT,
    default_hfov=DEFAULT_HFOV,
    default_vfov=DEFAULT_VFOV
):
    """
    Build partial equirect coverage in [-135..+135], [-30..+90]
    with NO alpha blending (last snapshot overwrites).
    """
    pano_map = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    pano_mask = np.zeros((out_height, out_width), dtype=np.uint8)
    total = len(image_list)

    for idx, (img, meta) in enumerate(zip(image_list, freeD_data_list)):
        # Flip vertically to fix orientation
        flipped_img = cv2.flip(img, 0)
        pan_center = meta.get('pan', 0.0)
        tilt_center = meta.get('tilt', 0.0)
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
# Place partial coverage in a full 360x180 equirect
# ---------------------------
def place_partial_in_full_equirect(partial_pano):
    """
    Place the partial coverage ([-135..+135], [-30..+90]) into a
    full 360x180 equirect ([-180..+180], [-90..+90]) of size FULL_W x FULL_H.
    The rest is black.
    """
    full_pano = np.zeros((FULL_H, FULL_W, 3), dtype=np.uint8)

    partial_h, partial_w = partial_pano.shape[:2]
    # partial domain: lon in [-135..+135], lat in [+90..-30]
    # full domain:    lon in [-180..+180], lat in [+90..-90]

    for py in range(partial_h):
        # lat in [+90..-30]
        lat_frac = py / (partial_h - 1)
        lat_deg = TILT_MAX - lat_frac * (TILT_MAX - TILT_MIN)  # from +90..-30

        if lat_deg < -90 or lat_deg > 90:
            continue
        lat_deg_frac = (lat_deg + 90.0) / 180.0
        fy = int(lat_deg_frac * FULL_H)
        if fy < 0 or fy >= FULL_H:
            continue

        row_ptr = partial_pano[py]
        for px in range(partial_w):
            lon_frac = px / (partial_w - 1)
            lon_deg = PAN_MIN + lon_frac * (PAN_MAX - PAN_MIN)  # [-135..+135]
            if lon_deg < -180 or lon_deg > 180:
                continue
            lon_deg_frac = (lon_deg + 180.0) / 360.0
            fx = int(lon_deg_frac * FULL_W)
            if fx < 0 or fx >= FULL_W:
                continue

            full_pano[fy, fx] = row_ptr[px]

    return full_pano

# ---------------------------
# Helper Functions
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

camera_ip = "192.168.12.8"
username = "admin"
password = "2Password2!"

def move_camera_absolute(pan_angle: float, tilt_angle: float) -> bool:
    pan_hex = convert_pan_angle(pan_angle)
    tilt_hex = convert_tilt_angle(tilt_angle)
    url = f"http://{camera_ip}/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&{pan_hex}&{tilt_hex}"
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
    1) Capture partial coverage in [-135..+135, -30..+90],
    2) Build partial equirect with NO alpha blending (last wins),
    3) Place partial coverage in a full 360x180 equirect (2:1), black for missing angles.
    """
    pan_values = np.arange(-135, 136, 45.0)
    tilt_values = np.arange(-30, 91, 30.0)
    positions = []
    for tilt in tilt_values:
        for pan in pan_values:
            positions.append({'pan': pan, 'tilt': tilt})
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

    # Save snapshots for debugging
    for i, im in enumerate(captured_images):
        cv2.imwrite(f"snapshot_{i}.jpg", im)

    # 1) Build partial equirect (270x120) with no alpha blending
    partial_pano = partial_stitch_images(
        captured_images, metadata_list,
        out_width=OUT_WIDTH, out_height=OUT_HEIGHT
    )
    cv2.imwrite("partial_panorama.jpg", partial_pano)
    cv2.imshow("Partial Equirect (270x120)", partial_pano)
    cv2.waitKey(500)

    # 2) Place partial coverage into full 360x180 equirect
    full_equi = place_partial_in_full_equirect(partial_pano)
    full_equi = cv2.flip(full_equi, 0)
    cv2.imwrite("full_equirect.jpg", full_equi)
    cv2.imshow("Full Equirect (2to1) with partial coverage", full_equi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_capture()
