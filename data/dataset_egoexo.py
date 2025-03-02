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

def my_collate(batch):
    images = [sample['image'] for sample in batch]
    condition = torch.stack([sample['cond'] for sample in batch])
    skeletons_window = torch.stack([sample['gt'] for sample in batch])
    head_offset = torch.stack([sample['offset'] for sample in batch])
    aria_window = torch.stack([sample['aria_future'] for sample in batch])
    frames_window = [sample['t'] for sample in batch]
    flags_window = torch.stack([sample['visible'] for sample in batch])
    task = [sample['task'] for sample in batch]
    take_name = [sample['take_name'] for sample in batch]
    take_uid = [sample['take_uid'] for sample in batch]
    
    return {
        'image':images, 
        'cond': condition, 
        'gt': skeletons_window,
        'visible': flags_window,
        't': frames_window,
        'aria_future': aria_window,
        'offset':head_offset,
        'task':task,
        'take_name':take_name,
        'take_uid':take_uid,
        'sequence':take_uid}

class Dataset_EgoExo_images(Dataset):
    def __init__(self, opt, root="dataset", slice_window=21,
                image_folder="dataset/image_takes/aria_214", use_pseudo=False):
        super(Dataset_EgoExo_images,self).__init__()

        self.split = opt['split']
        self.root = root
        self.root_takes = os.path.join(root, "takes")
        self.root_poses = os.path.join(root, "annotations", "ego_pose", self.split, "body")
        self.split = opt['split']
        self.use_pseudo = opt['use_pseudo']
        self.use_rot = opt["use_rot"]
        self.coord = opt["coord"]
        self.use_aria = opt["use_aria"]
        self.future = opt["future"]
        if self.future:
            self.slice_window =  opt['window_size']
        else:
            self.slice_window =  opt['window_size']+1
        self.output = opt["output"]
        self.opt = opt
        self.pred_input = opt['pred_input']
        assert self.use_aria and self.use_rot, "Rotation just works with Aria input"
        
        self.future_frames = opt["future_frames"]

        manually_annotated_takes = os.listdir(os.path.join(self.root_poses,"annotation"))
        manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]

        #Some video takes folders are empty when downloading them from egoexo-4D and need to be removed.

        if opt['video_model']:
            video_takes = set(os.listdir(image_folder))  # Use a set for O(1) lookups

            manually_annotated_takes = [take for take in manually_annotated_takes if take in video_takes]

        manually_annotated_takes = os.listdir(os.path.join(self.root_poses,"annotation"))
        manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]


        if self.use_pseudo:
            psuedo_annotated_takes = os.listdir(os.path.join(self.root_poses,"automatic"))
            psuedo_annotated_takes = [take.split(".")[0] for take in psuedo_annotated_takes]
        

        cameras = os.listdir(self.root_poses.replace("body", "camera_pose"))
        self.metadata = json.load(open(os.path.join(self.root,"takes.json")))

        self.takes_uids = psuedo_annotated_takes if self.use_pseudo else manually_annotated_takes
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
        max_window = max(self.slice_window ,self.future_frames)
        
        for take_uid in tqdm(self.takes_metadata):
            trajectory = {}
            if take_uid+".json" in cameras:
                camera_json = json.load(open(os.path.join(self.root_poses.replace("body", "camera_pose"),take_uid+".json")))
                take_name = camera_json['metadata']['take_name']
                self.cameras[take_uid] = camera_json
                self.cameras[take_uid] = camera_json
                if not take_uid in manually_annotated_takes:
                    #print("Not in manually annotated")
                    no_man +=1
                if self.use_pseudo and take_uid in psuedo_annotated_takes:
                    pose_json = json.load(open(os.path.join(self.root_poses,"automatic",take_uid+".json")))
                    if (len(pose_json)-self.future_frames > (self.slice_window)) and self.split == "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        if len(traj)-self.future_frames > (self.slice_window):
                            self.poses[take_uid] = ann
                            self.trajectories[take_uid] = traj
                    elif self.split != "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        self.poses[take_uid] = ann
                        self.trajectories[take_uid] = traj                
                elif take_uid in manually_annotated_takes:
                    pose_json = json.load(open(os.path.join(self.root_poses,"annotation",take_uid+".json")))
                    if (len(pose_json)-self.future_frames > (self.slice_window)) and self.split == "train":
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        if len(traj)-self.future_frames > (self.slice_window):
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
        try:
            capture_frames =  list(pose.keys())
            if self.future:
                frames_idx =  random.randint(self.slice_window, len(capture_frames)-self.future_frames)
            else:
                frames_idx =  random.randint(self.slice_window, len(capture_frames)-1)

            frames_window = capture_frames[frames_idx-self.slice_window: frames_idx]
            if self.future:
                frames_window_future = capture_frames[frames_idx: frames_idx+self.future_frames]
        except:
            breakpoint()

        images_window = []
        skeletons_window = []
        flags_window = []
        t_window = []
        aria_window = []
        #choose images
        skeletons_window_future = []
        flags_window_future = [] 
        aria_window_future = [] 

        for frame in frames_window:
            t_window.append(int(frame))
            skeleton = pose[frame][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory[frame])
            if self.video_model:
                image_path =  os.path.join(self.root,self.image_folder,take_uid, f"{frame}.png")
                image =  Image.open(image_path)
                if self.image_transforms is not None:
                    image =  self.image_transforms(image)
                images_window.append(image)
                
        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        flags_window =  torch.Tensor(np.array(flags_window))
        aria_window =  torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1,17,1)
        
        for frame in frames_window_future:
            skeleton = pose[frame][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window_future.append(skeleton)
            flags_window_future.append(flags)
            aria_window_future.append(aria_trajectory[frame])

        skeletons_window_future =  torch.Tensor(np.array(skeletons_window_future))
        flags_window_future =  torch.Tensor(np.array(flags_window_future))
        aria_window_future =  torch.Tensor(np.array(aria_window_future))


        if self.video_model:
            images_window =  torch.stack(images_window)
        
        if self.single_joint:
            if not self.use_aria:
                condition =  head_offset[:,0,:]
                skeletons_window[:,4,:]=skeletons_window[:,4,:]+head_offset[:,0,:]
                head_offset[:,4,:]=head_offset[:,4,:]-head_offset[:,4,:]
            else:
                if self.use_rot:
                    
                    if self.output=='aria':
                        condition =  aria_window[:] # Takes both trans and rot
                    else:
                        condition = torch.cat((skeletons_window.reshape(skeletons_window.shape[0],-1), aria_window),1)
                else:
                    condition =  aria_window[:,0:3] # from 0:3 to just translation vector
                    skeletons_window_future = torch.cat((skeletons_window_future, aria_window_future[:,:3].unsqueeze(1)),1) # add the translation vector to the skeleton
        else:
            condition = skeletons_window[:,[0,10, 2],:] #left wrist, right wrist, nose
        
        task = torch.tensor(self.takes_metadata[take_uid]['task_id'])
        take_name = self.takes_metadata[take_uid]['root_dir']
        
        if not self.video_model:
            return {'cond': condition, 
                    'gt': skeletons_window_future if self.output=='skeleton' else aria_window_future,
                    'visible': flags_window_future,
                    't': frames_window,#t_window,
                    'aria_future': aria_window_future,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid}
        else:
            return {'image':images_window,
                    'cond': condition, 
                    'gt': skeletons_window_future if self.output=='skeleton' else aria_window_future,
                    'visible': flags_window_future,
                    't': frames_window,#t_window,
                    'aria_future': aria_window_future,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid}
    
    def __len__(self):
        return len(self.poses)


class Dataset_EgoExo_images_test(Dataset):
    def __init__(self, opt, root="dataset", slice_window=21,
                image_folder= "dataset/image_takes/aria_214", use_pseudo=False):
        super(Dataset_EgoExo_images_test,self).__init__()
        self.root = root
        self.split = opt['split']
        self.root_poses = os.path.join(root, "annotations", "ego_pose", self.split, "body")
        self.use_pseudo = opt['use_pseudo']
        self.use_rot = opt["use_rot"]
        self.coord = opt["coord"]
        self.split = opt['split']
        self.use_aria = opt["use_aria"]
        self.future = opt["future"]
        self.output = opt["output"]
        self.opt = opt
        self.pred_input = opt['pred_input']
        assert self.use_aria and self.use_rot, "Rotation just works with Aria input"
        self.slice_window =  opt['window_size']
        self.future_frames = opt["future_frames"]

        # load sequences paths
        manually_annotated_takes = os.listdir(os.path.join(self.root_poses,"annotation"))
        manually_annotated_takes = [take.split(".")[0] for take in manually_annotated_takes]

        #Some video takes folders are empty when downloading them from egoexo-4D and need to be removed.

        if opt['video_model']:
            video_takes = set(os.listdir(image_folder))  # Use a set for O(1) lookups

            manually_annotated_takes = [take for take in manually_annotated_takes if take in video_takes]
        if self.use_pseudo:
            psuedo_annotated_takes = os.listdir(os.path.join(self.root_poses,"automatic"))
            psuedo_annotated_takes = [take.split(".")[0] for take in psuedo_annotated_takes]
        
        cameras = os.listdir(self.root_poses.replace("body", "camera_pose"))
        self.metadata = json.load(open(os.path.join(self.root,"takes.json")))


        self.takes_uids = psuedo_annotated_takes if self.use_pseudo else manually_annotated_takes
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
                if self.use_pseudo and take_uid in psuedo_annotated_takes:
                    with open(os.path.join(self.root_poses,"automatic",take_uid+".json")) as file:
                        pose_json = json.load(file)
                    if (len(pose_json)-self.future_frames >= (self.slice_window)):
                        ann, traj = self.translate_poses(pose_json, camera_json, self.coord, self.use_rot)
                        self.poses[take_uid] = ann
                        self.trajectories[take_uid] = traj
                elif take_uid in manually_annotated_takes:
                    pose_json = json.load(open(os.path.join(self.root_poses,"annotation",take_uid+".json")))
                    if (len(pose_json)-self.future_frames >= (self.slice_window)):
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

    def translate_poses_pred(self,anno, cams):
        for key in cams[list(cams.keys())[0]].keys():
            if "aria" in key:
                aria_key =  key
                break
        first = next(iter(anno))
        current_cam =  cams[first]
        T_world_camera = np.eye(4)
        T_world_camera[:3, :] = np.array(current_cam[aria_key]['camera_extrinsics'])
        for frame in anno:
            current_anno = anno[frame]
            current_cam_ =  cams[frame]
            T_world_camera_ = np.eye(4)
            T_world_camera_[:3, :] = np.array(current_cam_[aria_key]['camera_extrinsics'])
            for idx in range(len(current_anno)):
                joints = current_anno[idx]
                joint4d = np.ones(4)
                joint4d[:3] = np.array(joints)
                new_joint4d = np.linalg.inv(T_world_camera_).dot(joint4d)
                new_joint4d = T_world_camera.dot(new_joint4d)
                current_anno[idx] = list(new_joint4d[:3])
            anno[frame] = current_anno

        return anno

    def __getitem__(self, index):


        take_uid = self.poses_takes_uids[index]
        
        pose = self.poses[take_uid]
        aria_trajectory =  self.trajectories[take_uid]
        capture_frames =  list(pose.keys())

        images_window = []
        skeletons_window = []
        flags_window = []
        t_window = []
        aria_window = []

        for frame in capture_frames:
            t_window.append(int(frame))
            skeleton = pose[frame][0]["annotation3D"]
            skeleton, flags = self.parse_skeleton(skeleton)
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory[frame])
            if self.video_model and not self.future:
                image_path = os.path.join(self.root,self.image_folder,take_uid, f"{frame}.png")
                image =  Image.open(image_path)
                if self.image_transforms is not None:
                    image =  self.image_transforms(image)
                images_window.append(image)
            elif self.video_model:
                image_path =  os.path.join(self.root,self.image_folder,take_uid, f"{frame}.png")
                images_window.append(image_path)      
        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        flags_window =  torch.Tensor(np.array(flags_window))
        aria_window =  torch.Tensor(np.array(aria_window))
        head_offset = aria_window.unsqueeze(1).repeat(1,17,1)
 
        if self.video_model and not self.future:
            images_window =  torch.stack(images_window)
        
        if self.pred_input:
            json_files = glob.glob(os.path.join(self.opt['route'],'*pred.json'))
            for file in json_files:
                pred_all = json.load(open(file))
                if take_uid in pred_all.keys():
                    break
                
            preds = pred_all[take_uid]['body']
            with open(f'/home/mcescobar/EgoExo/annotations/ego_pose/hand/camera_pose/{take_uid}.json') as file:
                camera_json = json.load(file)
            preds_3D = self.translate_poses_pred(preds,camera_json)
            pred_values = np.array(list(preds_3D.values()))
            skeletons_window = torch.Tensor(pred_values)
        if self.single_joint:
            if not self.use_aria:
                condition =  head_offset[:,0,:]
                skeletons_window[:,4,:]=skeletons_window[:,4,:]+head_offset[:,0,:]
                head_offset[:,4,:]=head_offset[:,4,:]-head_offset[:,4,:]
            else:
                if self.use_rot:
                    if self.output=='aria':
                        condition =  aria_window[:] # Takes both trans and rot
                    else:
                        condition = torch.cat((skeletons_window.reshape(skeletons_window.shape[0],-1), aria_window),1)
                else:
                    condition =  aria_window[:,0:3] # from 0:3 to just translation vector
                    
        else:
            condition = skeletons_window[:,[0,10, 2],:] #left wrist, right wrist, nose
        
        task = torch.tensor(self.takes_metadata[take_uid]['task_id'])
        take_name = self.takes_metadata[take_uid]['root_dir']
        
        if not self.video_model:
            return {'cond': condition, 
                    'gt': skeletons_window,
                    'visible': flags_window,
                    't': t_window,#t_window,
                    'aria_future': condition,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid,
                    'sequence':take_uid}
        else:
            return {'image':images_window,
                    'cond': condition, 
                    'gt': skeletons_window,
                    'visible': flags_window,
                    't': t_window,#t_window,
                    'aria_future': condition,
                    'offset':head_offset,
                    'task':task,
                    'take_name':take_name,
                    'take_uid':take_uid,
                    'sequence':take_uid}
    
    def __len__(self):
        return len(self.poses)

if __name__=="__main__":
    image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = Dataset_EgoExo({"split":"train", "single_joint":False, "video_model":False, "use_pseudo": True},root="/media/SSD7/cdforigua/EgoExo")
    dataset.__getitem__(0)
