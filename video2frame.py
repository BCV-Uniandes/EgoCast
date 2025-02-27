import os
import json
import tqdm
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing.dummy as mp 

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_images(args):
    idx, root_poses, root_takes, phase, progress_bar = args
    folder = "annotation"

    image_output_path = f"dataset/image_takes/aria_214"
    os.makedirs(image_output_path, exist_ok=True)

    takes = os.listdir(os.path.join(root_poses, folder))
    take = takes[idx]
    take_id = take.split(".")[0]
    take_folder = os.path.join(image_output_path, take_id)
    os.makedirs(take_folder, exist_ok=True)

    take_json_path = os.path.join(root_poses, folder, take)
    take_json = json.load(open(take_json_path))

    takes_json_path = "dataset/takes.json"  # Adjust the path if necessary

    # Load takes.json
    with open(takes_json_path, "r") as f:
        takes_data = json.load(f)

    take_name_dict = {entry["take_name"]: entry["take_uid"] for entry in takes_data}
    take_name = next((k for k, v in take_name_dict.items() if v == take_id), None)

    if not take_name:
        print(f"Take name not found for {take_id}")
        return

    if not os.path.exists(os.path.join(root_takes, take_name, "frame_aligned_videos")):
        print(f"No video found for take: {take_id}")
        
        return

    aria_file = None
    video_dir = os.path.join(root_takes, take_name, "frame_aligned_videos", "downscaled", "448")
    
    for file in os.listdir(video_dir):
        if file.endswith("214-1.mp4"):
            aria_file = file
            break

    if not aria_file:
        #print(f"No matching video file found in {video_dir}")
        print(take_id)
        return

    video_path = os.path.join(video_dir, aria_file)
    video_frames = extract_frames(video_path)

    for frame in take_json:
        try:
            frame_name = frame + ".png"
            frame_path = os.path.join(take_folder, frame_name)
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, video_frames[int(frame)])
        except Exception as e:
            print(f"Error in frame {frame}: {e}")
            continue

    progress_bar.update(1)  # Update the tqdm progress bar

def main():
    root_takes = "dataset/takes"
    base_root_poses = "dataset/annotations/ego_pose"

    phases = ["train", "test", "val"]

    for phase in phases:
        root_poses = os.path.join(base_root_poses, phase, "body")
        if not os.path.exists(root_poses):
            print(f"Skipping {phase}, directory does not exist: {root_poses}")
            continue

        takes = os.listdir(os.path.join(root_poses, "annotation"))
        
        print(f"Processing {phase}: {len(takes)} takes")

        with tqdm.tqdm(total=len(takes), desc=f"Extracting frames for {phase}") as progress_bar:
            p = mp.Pool(10)  # Adjust the number of workers as needed
            p.map(extract_images, [(idx, root_poses, root_takes, phase, progress_bar) for idx in range(len(takes))])
            p.close()
            p.join()

if __name__ == "__main__":
    main()
