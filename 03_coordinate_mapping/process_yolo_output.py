import argparse
import os
import re
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import yaml


'''
Converts the per-frame label files written by detect_and_track.py (--track --save-txt)
into one table with timestamps and real-world (x, y) coordinates in metres.

input folder layout (one sub-folder per video / "scene"):

    input_dir/
    ├── scene_1/
    │   ├── video.mp4        # undistorted video (only used to read the frame size)
    │   ├── ts.txt           # start time of the video: "YYYY-MM-DD HH:MM:SS" or "HH:MM:SS"
    │   └── labels/          # *_<frame>.txt files from detect_and_track.py
    └── scene_2/ ...

output: CSV with columns
    classname, obj_id, classid, x, y, w, h, frame, scene, ts, x_px_coord, y_px_coord, x_3d, y_3d

usage: python process_yolo_output.py --input-dir ./sample_input --fps 30 --config ../config/camera.yaml --output coords_3d.csv
'''


def load_camera(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    i = cfg['intrinsics']
    K = np.array([[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1]], dtype=np.float64)
    image_points = np.array(cfg['image_points'], dtype=np.float32)
    world_points = np.array(cfg['world_points'], dtype=np.float32)
    return K, image_points, world_points


def estimate_pose(K, image_points, world_points):
    """Rotation matrix and translation vector of the camera w.r.t. the ground plane."""
    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points, image_points, K, None)  # video is already undistorted -> no dist coeffs
    if not success:
        raise RuntimeError("solvePnP failed, check image_points / world_points")
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix, translation_vector


def pixels_to_ground(u, v, K, rotation_matrix, translation_vector):
    """Back-project pixel coordinates onto the Z = 0 world plane. Returns X, Y in world units."""
    uv = np.vstack([u, v, np.ones_like(u)])  # 3 x N
    R_inv = np.linalg.inv(rotation_matrix)
    K_inv = np.linalg.inv(K)

    ray = R_inv @ K_inv @ uv                       # 3 x N
    cam_origin = R_inv @ translation_vector        # 3 x 1
    s = cam_origin[2, 0] / ray[2]                  # scale so that Z = 0
    P = R_inv @ (s * (K_inv @ uv) - translation_vector)
    return P[0], P[1]


def parse_start_time(text):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised start time in ts.txt: '{text}'")


def read_yolo_txt(file_path):
    # Extract frame number from the file name, e.g. video_123.txt -> 123
    match = re.search(r'_([0-9]+)\.txt$', os.path.basename(file_path))
    frame = int(match.group(1)) if match else None

    data = []
    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split()
            if len(components) != 7:
                continue
            classname, obj_id, classid, x, y, w, h = components
            data.append([classname, int(float(obj_id)), int(float(classid)),
                         float(x), float(y), float(w), float(h), frame])
    return data


def process_yolo_output(input_dir, fps, config_path, units_per_metre=1000):
    K, image_points, world_points = load_camera(config_path)
    rotation_matrix, translation_vector = estimate_pose(K, image_points, world_points)

    dfs = []

    # Iterate through each scene folder in the directory
    for folder_name in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label_folder_path = os.path.join(folder_path, "labels")
        ts_file_path = os.path.join(folder_path, "ts.txt")
        video_file_path = next((os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                if f.lower().endswith((".mp4", ".avi", ".mov"))), None)

        if not (os.path.exists(label_folder_path) and os.path.exists(ts_file_path) and video_file_path):
            print(f"Skipping '{folder_name}': needs labels/, ts.txt and a video file")
            continue

        # get the frame size
        cap = cv2.VideoCapture(video_file_path)
        framesize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        # Read the start timestamp from ts.txt
        with open(ts_file_path, 'r') as ts_file:
            start_timestamp = parse_start_time(ts_file.read().strip())

        rows = []
        for filename in os.listdir(label_folder_path):
            if filename.endswith(".txt"):
                rows.extend(read_yolo_txt(os.path.join(label_folder_path, filename)))
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=['classname', 'obj_id', 'classid', 'x', 'y', 'w', 'h', 'frame'])
        df['scene'] = folder_name  # used to differentiate the videos

        # timestamp of each frame
        df['ts'] = [start_timestamp + timedelta(seconds=f / fps) for f in df['frame']]

        # normalised box centre -> pixel coordinates
        df['x_px_coord'] = df['x'] * framesize[0]
        df['y_px_coord'] = df['y'] * framesize[1]

        # pixel -> real-world coordinates (metres)
        X, Y = pixels_to_ground(df['x_px_coord'].to_numpy(), df['y_px_coord'].to_numpy(),
                                K, rotation_matrix, translation_vector)
        df['x_3d'] = X / units_per_metre
        df['y_3d'] = Y / units_per_metre

        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No usable scenes found in {input_dir}")

    final_df = pd.concat(dfs, ignore_index=True).sort_values(['scene', 'frame', 'obj_id'])
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map tracked detections to real-world coordinates.')
    parser.add_argument('--input-dir', default='./sample_input', help='folder with one sub-folder per scene')
    parser.add_argument('--fps', type=float, default=30, help='frame rate of the videos')
    parser.add_argument('--config', default='../config/camera.yaml', help='camera calibration file')
    parser.add_argument('--output', default='coords_3d.csv', help='output CSV file')
    opt = parser.parse_args()

    df = process_yolo_output(opt.input_dir, opt.fps, opt.config)
    df.to_csv(opt.output, index=False)
    print(f"Saved {len(df)} rows to {opt.output}")
