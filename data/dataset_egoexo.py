import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
import multiprocessing
import random
import threading
import glob
import skimage.io as io
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
#from mpi4py import MPI
from scipy.spatial.transform import Rotation as R

random.seed(1)

def my_collate_multiprocessing(batch):

    images = [sample['image'] for sample in batch]
    condition = torch.stack([sample['cond'] for sample in batch])
    skeletons_window = torch.stack([sample['gt'] for sample in batch])
    head_offset = torch.stack([sample['offset'] for sample in batch])
    aria_window = torch.stack([sample['aria'] for sample in batch])
    frames_window = [sample['t'] for sample in batch]
    flags_window = torch.stack([sample['visible'] for sample in batch])
    take_name = [sample['take_name'] for sample in batch]
    take_uid = [sample['take_uid'] for sample in batch]
    
    return {
        'image':images, 
        'cond': condition, 
        'gt': skeletons_window,
        'visible': flags_window,
        't': frames_window,
        'aria': aria_window,
        'offset':head_offset,
        'take_name':take_name,
        'take_uid':take_uid}


def my_collate(batch):
    images = [sample['image'] for sample in batch]
    conds = torch.stack([sample['cond'] for sample in batch])
    gts = torch.stack([sample['gt'] for sample in batch])
    offsets = torch.stack([sample['offset'] for sample in batch])
    activities = [sample['activity'] for sample in batch]
    return {
        'image': images,
        'cond': conds,
        'gt': gts,
        'offset': offsets,
        'activity': activities}

class Dataset_EgoExo_images(Dataset):
    def __init__(self, opt, root="dataset", slice_window=21,
                image_folder="dataset/image_takes/aria_214", use_pseudo=False):
        super(Dataset_EgoExo_images,self).__init__()
        self.root = root
        self.split = opt['split']

        
        self.root_poses = os.path.join(root, "annotations", "ego_pose", self.split, "body")
        self.use_pseudo = opt['use_pseudo']
        self.use_rot = opt["use_rot"]
        self.coord = opt["coord"]
        self.slice_window =  slice_window
        # load sequences paths

        
        manually_annotated_takes = os.listdir(os.path.join(self.root_poses,"annotation"))
        manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]

        #Some video takes folders are empty when downloading them from egoexo-4D and need to be removed.

        if opt['video_model']:
            video_takes = set(os.listdir(image_folder))  # Use a set for O(1) lookups

            manually_annotated_takes = [take for take in manually_annotated_takes if take in video_takes]

        
        if self.use_pseudo:
            pseudo_annotated_takes = os.listdir(os.path.join(self.root_poses,"automatic"))
            pseudo_annotated_takes = [take.split(".")[0] for take in pseudo_annotated_takes]
        
        cameras = os.listdir(self.root_poses.replace("body", "camera_pose"))
        self.metadata = json.load(open(os.path.join(self.root,"takes.json")))
        
        self.takes_uids = pseudo_annotated_takes if self.use_pseudo else manually_annotated_takes
        self.takes_metadata = {}
        img_list = os.listdir(image_folder)
        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and 'bouldering' not in take_temp['take_name']:
                self.takes_metadata[take_uid] = take_temp

        self.poses = {}
        self.trajectories = {}
        self.cameras = {}
        manually = 0
        no_man = 0
        no_cam = 0
        no_cam_list = []
        for take_uid in tqdm(self.takes_metadata):
            trajectory = {}

            if take_uid+".json" in cameras:
                camera_json = json.load(open(os.path.join(self.root_poses.replace("body", "camera_pose"),take_uid+".json")))
                take_name = camera_json['metadata']['take_name']
                self.cameras[take_uid] = camera_json
                if not take_uid in manually_annotated_takes:
         
                    no_man +=1
                if self.use_pseudo and take_uid in pseudo_annotated_takes:
                    pose_json = json.load(open(os.path.join(self.root_poses,"automatic",take_uid+".json")))
                    if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        if len(traj) > (self.slice_window +2):
                            self.poses[take_uid] = ann
                            self.trajectories[take_uid] = traj
                    elif self.split != "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        self.poses[take_uid] = ann
                        self.trajectories[take_uid] = traj
                elif take_uid in manually_annotated_takes:
                    pose_json = json.load(open(os.path.join(self.root_poses,"annotation",take_uid+".json")))
                    if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        if len(traj) > (self.slice_window +2):
                            self.poses[take_uid] = ann
                            self.trajectories[take_uid] = traj
                    elif self.split != "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        self.poses[take_uid] = ann
                        self.trajectories[take_uid] = traj

            else:
                #print("No take uid {} in camera poses".format(take_uid))
                no_cam += 1
                no_cam_list.append(take_uid)
        new_pose = {}
        for pose in self.poses:
            #if(len(self.poses[pose]))>self.slice_window+2:
            new_pose[pose] = self.poses[pose]
        self.poses = new_pose
        self.joint_idxs = [i for i in range(17)]
        self.joint_names = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
        self.single_joint = opt['single_joint']
        self.video_model = opt['video_model']
        self.image_transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.image_folder =  image_folder
        self.poses_takes_uids = list(self.poses.keys())
        

        print('Dataset lenght: {}'.format(len(self.poses)))
        print('Split: {}'.format(self.split))
        print('No Manually: {}'.format(no_man))
        print('No camera: {}'.format(no_cam))
        print('No camera list: {}'.format(no_cam_list))

    def translate_poses(self, anno, cams, coord,use_rot=False):
        trajectory = {}
        to_remove = []
        for key in cams.keys():
            if "aria" in key:
                aria_key =  key
                break
        first = next(iter(anno))
        first_cam =  cams[aria_key]['camera_extrinsics'][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in anno:
            try:
                current_anno = anno[frame]
                current_cam =  cams[aria_key]['camera_extrinsics'][frame]
                T_world_camera_ = np.eye(4)
                T_world_camera_[:3, :] = np.array(current_cam)
                
                if coord == 'global':
                    T_world_camera = np.linalg.inv(T_world_camera_)
                elif coord == 'aria':
                    T_world_camera = np.dot(T_first_camera,np.linalg.inv(T_world_camera_))
                else:
                    T_world_camera = T_world_camera_
                assert len(current_anno) != 0 
                for idx in range(len(current_anno)):
                    joints = current_anno[idx]["annotation3D"]
                    for joint_name in joints:
                        joint4d = np.ones(4)
                        joint4d[:3] = np.array([joints[joint_name]["x"], joints[joint_name]["y"], joints[joint_name]["z"]])
                        if coord == 'global':
                            new_joint4d = joint4d
                        elif coord == 'aria':
                            new_joint4d = T_first_camera.dot(joint4d)
                        else:
                            new_joint4d = T_world_camera_.dot(joint4d) #The skels always stay in 0,0,0 wrt their camera frame
                        joints[joint_name]["x"] = new_joint4d[0]
                        joints[joint_name]["y"] = new_joint4d[1]
                        joints[joint_name]["z"] = new_joint4d[2]
                    current_anno[idx]["annotation3D"] = joints
                traj = T_world_camera[:3,3]
                rot = R.from_matrix(T_world_camera[:3,:3])
                q_rot = rot.as_quat()
                if use_rot:
                    traj = np.concatenate([traj,q_rot])
                trajectory[frame] = traj
            except:
                to_remove.append(frame)
            anno[frame] = current_anno
        keys_old = list(anno.keys())
        for frame in keys_old:
            if frame in to_remove:
                del anno[frame]
        return anno, trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"]==uid:
                return take
    def get_metadata_take_from_name(self, name):
        for take in self.metadata:
            if take["_s3_root_dir"].split("/")[-1]==name:
                return take
    def parse_skeleton(self, skeleton):
        poses = []
        flags = []
        keypoints = skeleton.keys()
        for keyp in self.joint_names:
            if keyp in keypoints:
                flags.append(1) #visible
                poses.append([skeleton[keyp]['x'], skeleton[keyp]['y'], skeleton[keyp]['z']]) #visible
            else:
                flags.append(0) #not visible
                poses.append([-1,-1,-1]) #not visible
        return poses, flags

    def __getitem__(self, index):
        take_uid = self.poses_takes_uids[index]
        pose = self.poses[take_uid]
        aria_trajectory =  self.trajectories[take_uid]

        capture_frames =  list(pose.keys())

        if self.split == "train":
            frames_idx = random.randint(self.slice_window, len(capture_frames)-1)
            frames_window = capture_frames[frames_idx-self.slice_window: frames_idx]
        else:
            frames_window = capture_frames
        images_window = []
        skeletons_window = []
        flags_window = []
        t_window = []
        aria_window = []
        #choose images

        for frame in frames_window:
            t_window.append(int(frame))
            skeleton = pose[frame][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory[frame])
            
            if self.video_model:
                image_path =  os.path.join(self.image_folder,take_uid, f"{frame}.png")
  
                image =  Image.open(image_path)
                if self.image_transforms is not None:
                    image =  self.image_transforms(image)
                images_window.append(image)
    
        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        flags_window =  torch.Tensor(np.array(flags_window))
        aria_window =  torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1,17,1)
        skeletons_window = skeletons_window #- head_offset

        if self.video_model:
            images_window =  torch.stack(images_window)
        
        if self.single_joint:
            condition =  aria_window
        else:
            condition = skeletons_window[:,[0,10, 2],:] #left wrist, right wrist, nose
        task = torch.tensor(self.takes_metadata[take_uid]['task_id'])
        take_name = self.takes_metadata[take_uid]['root_dir']
        
        if not self.video_model:
            return {'cond': condition, 
                    'gt': skeletons_window,
                    'visible': flags_window,
                    't': frames_window,#t_window,
                    'aria': aria_window,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid}
        else:
            return {'image':images_window,
                    'cond': condition, 
                    'gt': skeletons_window,
                    'visible': flags_window,
                    't': frames_window,#t_window,
                    'aria': aria_window,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid}
    
    def __len__(self):
        return len(self.poses)


class Dataset_EgoExo_images_multi(Dataset):
    def __init__(self, opt, root="dataset", slice_window=21,
                image_folder="dataset/image_takes/aria_214", use_pseudo=False):
        
        super(Dataset_EgoExo_images_multi,self).__init__()

        self.root = root
        #self.root_takes = os.path.join(root, "takes")
        self.split = opt['split']

        
        self.root_poses = os.path.join(root, "annotations", "ego_pose", self.split, "body")
        self.use_pseudo = opt['use_pseudo']
        self.use_rot = opt["use_rot"]
        self.coord = opt["coord"]
        self.slice_window =  slice_window
        # load sequences paths

        
        manually_annotated_takes = os.listdir(os.path.join(self.root_poses,"annotation"))
        manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]

        #Some video takes folders are empty when downloading them from egoexo-4D and need to be removed.

        if opt['video_model']:
            video_takes = set(os.listdir(image_folder))  # Use a set for O(1) lookups

            manually_annotated_takes = [take for take in manually_annotated_takes if take in video_takes]

        if self.use_pseudo:
            pseudo_annotated_takes = os.listdir(os.path.join(self.root_poses,"automatic"))
            pseudo_annotated_takes = [take.split(".")[0] for take in pseudo_annotated_takes]
        
        cameras = os.listdir(self.root_poses.replace("body", "camera_pose"))
        
        self.metadata = json.load(open(os.path.join(self.root,"takes.json")))

        #distribution_path = "/media/SSD5/mcescobar/EgoExo/annotations/egoexo_split_latest_train_val_test.csv"
        #self.distribution = pd.read_csv(distribution_path)
        #self.distribution = self.distribution[self.distribution["split"]==self.split.upper()]
        #self.takes_uids = self.distribution["take_uid"].values
        self.takes_uids = pseudo_annotated_takes if self.use_pseudo else manually_annotated_takes
        self.takes_metadata = {}
        img_list = os.listdir(image_folder)
        for take_uid in self.takes_uids:
            take_temp = self.get_metadata_take(take_uid)
            if take_temp and 'bouldering' not in take_temp['take_name']:
                self.takes_metadata[take_uid] = take_temp

        self.poses = {}
        self.trajectories = {}
        self.cameras = {}
        manually = 0
        no_man = 0
        no_cam = 0
        no_cam_list = []


        for take_uid in tqdm(self.takes_metadata):
            trajectory = {}

            if take_uid+".json" in cameras:
                camera_json = json.load(open(os.path.join(self.root_poses.replace("body", "camera_pose"),take_uid+".json")))
                take_name = camera_json['metadata']['take_name']
                self.cameras[take_uid] = camera_json
                if not take_uid in manually_annotated_takes:
                    #print("Not in manually annotated")
                    no_man +=1
                if self.use_pseudo and take_uid in pseudo_annotated_takes:
                    pose_json = json.load(open(os.path.join(self.root_poses,"automatic",take_uid+".json")))
                    if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        if len(traj) > (self.slice_window +2):
                            self.poses[take_uid] = ann
                            self.trajectories[take_uid] = traj
                    elif self.split != "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        self.poses[take_uid] = ann
                        self.trajectories[take_uid] = traj
                elif take_uid in manually_annotated_takes:
                    pose_json = json.load(open(os.path.join(self.root_poses,"annotation",take_uid+".json")))
                    if (len(pose_json) > (self.slice_window +2)) and self.split == "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        if len(traj) > (self.slice_window +2):
                            self.poses[take_uid] = ann
                            self.trajectories[take_uid] = traj
                    elif self.split != "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        self.poses[take_uid] = ann
                        self.trajectories[take_uid] = traj

            else:
                #print("No take uid {} in camera poses".format(take_uid))
                no_cam += 1
                no_cam_list.append(take_uid)
        new_pose = {}
        for pose in self.poses:
            #if(len(self.poses[pose]))>self.slice_window+2:
            new_pose[pose] = self.poses[pose]
        self.poses = new_pose
        self.joint_idxs = [i for i in range(17)] # 17 keypoints in total
        #self.joint_names = ['left-wrist', 'left-eye', 'nose', 'right-elbow', 'left-ear', 'left-shoulder', 'right-hip', 'right-ear', 'left-knee', 'left-hip', 'right-wrist', 'right-ankle', 'right-eye', 'left-elbow', 'left-ankle', 'right-shoulder', 'right-knee']
        self.joint_names = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
        self.single_joint = opt['single_joint']
        self.video_model = opt['video_model']
        self.image_transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.image_folder =  image_folder
        self.poses_takes_uids = list(self.poses.keys())
        
        print('Dataset lenght: {}'.format(len(self.poses)))
        print('Split: {}'.format(self.split))
        print('No Manually: {}'.format(no_man))
        print('No camera: {}'.format(no_cam))
        print('No camera list: {}'.format(no_cam_list))

    def translate_poses(self, anno, cams, coord,use_rot=False):
        trajectory = {}
        to_remove = []
        for key in cams.keys():
            if "aria" in key:
                aria_key =  key
                break
        first = next(iter(anno))
        first_cam =  cams[aria_key]['camera_extrinsics'][first]
        T_first_camera = np.eye(4)
        T_first_camera[:3, :] = np.array(first_cam)
        for frame in anno:
            try:
                current_anno = anno[frame]
                current_cam =  cams[aria_key]['camera_extrinsics'][frame]
                T_world_camera_ = np.eye(4)
                T_world_camera_[:3, :] = np.array(current_cam)
                
                if coord == 'global':
                    T_world_camera = np.linalg.inv(T_world_camera_)
                elif coord == 'aria':
                    T_world_camera = np.dot(T_first_camera,np.linalg.inv(T_world_camera_))
                else:
                    T_world_camera = T_world_camera_
                assert len(current_anno) != 0 
                for idx in range(len(current_anno)):
                    joints = current_anno[idx]["annotation3D"]
                    for joint_name in joints:
                        joint4d = np.ones(4)
                        joint4d[:3] = np.array([joints[joint_name]["x"], joints[joint_name]["y"], joints[joint_name]["z"]])
                        if coord == 'global':
                            new_joint4d = joint4d
                        elif coord == 'aria':
                            new_joint4d = T_first_camera.dot(joint4d)
                        else:
                            new_joint4d = T_world_camera_.dot(joint4d) #The skels always stay in 0,0,0 wrt their camera frame
                        joints[joint_name]["x"] = new_joint4d[0]
                        joints[joint_name]["y"] = new_joint4d[1]
                        joints[joint_name]["z"] = new_joint4d[2]
                    current_anno[idx]["annotation3D"] = joints
                traj = T_world_camera[:3,3]
                rot = R.from_matrix(T_world_camera[:3,:3])
                q_rot = rot.as_quat()
                if use_rot:
                    traj = np.concatenate([traj,q_rot])
                trajectory[frame] = traj
            except:
                to_remove.append(frame)
            anno[frame] = current_anno
        keys_old = list(anno.keys())
        for frame in keys_old:
            if frame in to_remove:
                del anno[frame]
        return anno, trajectory

    def get_metadata_take(self, uid):
        for take in self.metadata:
            if take["take_uid"]==uid:
                return take
    def get_metadata_take_from_name(self, name):
        for take in self.metadata:
            if take["_s3_root_dir"].split("/")[-1]==name:
                return take
    def parse_skeleton(self, skeleton):
        poses = []
        flags = []
        keypoints = skeleton.keys()
        for keyp in self.joint_names:
            if keyp in keypoints:
                flags.append(1) #visible
                poses.append([skeleton[keyp]['x'], skeleton[keyp]['y'], skeleton[keyp]['z']]) #visible
            else:
                flags.append(0) #not visible
                poses.append([-1,-1,-1]) #not visible
        return poses, flags

    def __getitem__(self, index):

        take_uid = self.poses_takes_uids[index]
        pose = self.poses[take_uid]
        aria_trajectory =  self.trajectories[take_uid]
        capture_frames =  list(pose.keys())
        if self.split == "train":
            frames_idx = random.randint(self.slice_window, len(capture_frames)-1)
            frames_window = capture_frames[frames_idx-self.slice_window: frames_idx]
        else:
            frames_window = capture_frames
        images_window = []
        skeletons_window = []
        flags_window = []
        t_window = []
        aria_window = []
        #choose images

        for frame in frames_window:
            t_window.append(int(frame))
            skeleton = pose[frame][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory[frame])
            if self.video_model:
                image_path =  os.path.join(self.root,self.image_folder,take_uid, f"{frame}.png")
                # image =  Image.open(image_path)
                # if self.image_transforms is not None:
                #     image =  self.image_transforms(image)
                images_window.append(image_path)
                
        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        flags_window =  torch.Tensor(np.array(flags_window))
        aria_window =  torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1,17,1)
        skeletons_window = skeletons_window #- head_offset

        if self.single_joint:
            condition =  aria_window
        else:
            condition = skeletons_window[:,[0,10, 2],:] #left wrist, right wrist, nose
        task = torch.tensor(self.takes_metadata[take_uid]['task_id'])
        take_name = self.takes_metadata[take_uid]['root_dir']
        
        if not self.video_model:
            return {'cond': condition, 
                    'gt': skeletons_window,
                    'visible': flags_window,
                    't': frames_window,#t_window,
                    'aria': aria_window,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid}
        else:
            return {'image':images_window,
                    'cond': condition, 
                    'gt': skeletons_window,
                    'visible': flags_window,
                    't': frames_window,#t_window,
                    'aria': aria_window,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid}
    
    def __len__(self):
        return len(self.poses)

