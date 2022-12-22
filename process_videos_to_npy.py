"""Script to save a .mp4 video file as npy vertices"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def video_to_npy(video_path) -> np.ndarray:
    """Given a path to a video (mp4), return the video as a np.array of images"""
    capture = cv2.VideoCapture(str(video_path))
    
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = np.empty((n_frames, frame_height, frame_width, 3), np.dtype('uint8'))

    frame_count = 0
    returned = True

    while (frame_count < n_frames  and returned):
        returned, video[frame_count] = capture.read()
        frame_count += 1

    capture.release()

    # returned cropped video sampled at 5 FPS
    return video[::6, 350:600:2, 275:525:2]


def convert_videos(videos_path, save_path):
    videos_path = Path(videos_path)
    save_path = Path(save_path)
    
    for video_file in videos_path.glob("*.mp4"):
        video = video_to_npy(video_file)

        root = str(video_file.stem)
        save_file = save_path / Path(root + ".npy")

        np.save(save_file, video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", type=str, default="vocaset/videos/")
    parser.add_argument("--save-path", type=str, default="vocaset/videos_npy/")
    args = parser.parse_args()
    convert_videos(args.videos_path, args.save_path)


if __name__ == "__main__":
    main()