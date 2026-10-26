import argparse

import cv2
import numpy as np
import yaml


'''
input: original (distorted) video + camera calibration file;
output: rectified video for the following analysis steps;

usage: python undistort.py --input input_video.mp4 --output undistorted.mp4 --config ../config/camera.yaml
'''


def load_calibration(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    i = cfg['intrinsics']
    K = np.array([[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1]], dtype=np.float32)  # camera matrix
    dist_coeffs = np.array(cfg['dist_coeffs'], dtype=np.float32)
    return K, dist_coeffs


def undistort_video(input_path, output_path, K, dist_coeffs, fps=30, show=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # New camera matrix for undistortion
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (width, height), 1, (width, height))
    x, y, w, h = roi

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = cv2.undistort(frame, K, dist_coeffs, None, new_K)
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]

        out.write(undistorted_frame)

        if show:
            cv2.imshow('Undistorted Frame', undistorted_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # press 'Esc' to exit
                break

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove lens distortion from a video.')
    parser.add_argument('--input', required=True, help='path to the original video')
    parser.add_argument('--output', default='undistorted_output_video.mp4', help='path of the rectified video')
    parser.add_argument('--config', default='../config/camera.yaml', help='camera calibration file')
    parser.add_argument('--fps', type=float, default=30, help='frame rate of the output video')
    parser.add_argument('--show', action='store_true', help='display frames while processing')
    opt = parser.parse_args()

    K, dist_coeffs = load_calibration(opt.config)
    undistort_video(opt.input, opt.output, K, dist_coeffs, opt.fps, opt.show)
