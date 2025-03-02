import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

def save_videos(gt,pr,aria,video_dir):
    # get all available skeletons in a sequence
    joint_idxs = [ 0 ,1,  2,  3,  4,  5,  6,  7,  8, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50]
    dict_joints = {k: v for v, k in enumerate(joint_idxs)}
    joint_connections = [(4, 3), (3, 2), (2, 1), (1, 0), (0, 43), (43, 44), (44, 45), (45, 46), (0, 47), (47, 48), (48, 49), (49, 50), (2, 5), (5, 6), (6, 7), (7, 8), (2, 24), (24, 25), (25, 26), (26, 27)]
    space = np.array([[-4.18003368,-0.09262197,-4.09477139], [-4.18003368,-0.09262197,-4.09477139],[ 5.58601093,-0.09262197,9.02135086], [5.58601093,-0.09262197,-4.09477139],   [-4.18003368,8.35954094,9.02135086],  [5.58601093,8.35954094,9.02135086], [-4.18003368,8.35954094,-4.09477139], [5.58601093,8.35954094,-4.09477139]])
    # draw skeleton
    if not os.path.exists(video_dir):
                os.makedirs(video_dir) 
    frames = min(1800,len(gt))
    for idx in tqdm(range(0,frames)):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i in range(0, len(joint_connections)):
            gt_1 = gt[idx][dict_joints[joint_connections[i][0]]]
            gt_2 = gt[idx][dict_joints[joint_connections[i][1]]]
            pr_1 = pr[idx][dict_joints[joint_connections[i][0]]]
            pr_2 = pr[idx][dict_joints[joint_connections[i][1]]]
            ax.scatter([gt_1[0], gt_2[0]], [gt_1[1], gt_2[1]], [gt_1[2], gt_2[2]],alpha=0.5,c='green')
            ax.plot([gt_1[0], gt_2[0]], [gt_1[1], gt_2[1]], [gt_1[2], gt_2[2]],alpha=0.5,c='green')
            ax.scatter([pr_1[0], pr_2[0]], [pr_1[1], pr_2[1]], [pr_1[2], pr_2[2]],alpha=1,c='blue')
            ax.plot([pr_1[0], pr_2[0]], [pr_1[1], pr_2[1]], [pr_1[2], pr_2[2]],alpha=1,c='blue')
        #breakpoint()
        aria_1=aria[idx]
        ax.scatter(aria_1[0],aria_1[1], aria_1[2],alpha=0.5,c='red')
        #ax.scatter(space.T[0],space.T[1],space.T[2],alpha=0.5,c='black')
        #ax.voxels(space)
        plt.xlim([-2,2])
        plt.ylim([0,2])
        ax.set_zlim(0,3)
        ax.view_init(elev=111, azim=-90)
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.savefig(os.path.join(video_dir,str(idx).zfill(5)+'.png'))
        plt.close()

route=os.path.join('results_singlejoint_aria_debug','AvatarPoseEstimation','videos')
files = os.listdir(route)
files = [a for a in files if ".npz" in a]
new_route = os.path.join('results_singlejoint_aria_debug','AvatarPoseEstimation','full_videos')
for f in files:
    container = np.load(os.path.join(route,f))
    gt = container['gt']
    pred = container['pr']
    timestamp = container['timestamp']
    error = container['error']
    aria = container['aria'][0]
    video_dir = os.path.join(route,f)[:-4]
    save_videos(gt,pred,aria,video_dir)
    frames = os.listdir(video_dir)
    frames.sort()
    for i,t in enumerate(timestamp[:len(frames)][:,0]):
        frame = frames[i]
        frame_path = os.path.join(video_dir,frame)
        img_path = os.path.join('/media/lambda001/SSD5/cdforigua/ADT/',f[:-4],'undistorted_images',t+'.png')
        images = [Image.open(x) for x in [frame_path,img_path]]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_video_dir = os.path.join(new_route,f)[:-4]
        if not os.path.exists(new_video_dir):
            os.makedirs(new_video_dir) 
        new_im.save(os.path.join(new_video_dir,frame))



