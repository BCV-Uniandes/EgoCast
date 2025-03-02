import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
import plotly.graph_objects as go
from projectaria_tools import utils
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinSkeletonProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
   bbox2d_to_image_coordinates,
   utils as adt_utils,
)
from projectaria_tools.core import data_provider, calibration
import multiprocessing
import random
import threading
import glob
import skimage.io as io
import torchvision.transforms as T
#from mpi4py import MPI

random.seed(1)
class Dataset_ADT_preloaded(Dataset):
    def __init__(self, opt, root_sequences="/media/lambda001/SSD5/cdforigua/ADT/", fold=1, distribution_folder="./data", skeleton_occlusions=False, slice_window=None,
                image_transforms=None, image_folder="/media/SSD7/cdforigua/windows_EgoPose"):
        super(Dataset_ADT_preloaded,self).__init__()
        self.root_sequences = root_sequences
        self.fold = opt['fold']
        self.skeleton_occlusions =skeleton_occlusions
        self.future = opt["future"]
        if self.future:
            self.slice_window =  opt['window_size']
        else:
            self.slice_window =  opt['window_size']+1
        self.use_aria = opt["use_aria"]
        self.use_rot = opt["use_rot"]
        self.output = opt["output"]
        assert self.use_aria and self.use_rot, "Rotation just works with Aria input"
        
        self.future_frames = opt["future_frames"]
        # load sequences paths
        self.sequences = json.load(open(os.path.join(distribution_folder, "distribution_skeleton_filter.json")))["fold1" if self.fold==1 else "fold2"]
        self.sequences =  [sequence.split("/")[-1] for sequence in self.sequences]
        self.stream_id = StreamId("214-1") #rgb_image
        self.joint_labels = AriaDigitalTwinSkeletonProvider.get_joint_labels()
        self.joint_connections = AriaDigitalTwinSkeletonProvider.get_joint_connections()
        self.joint_idxs = [ 0 ,1,  2,  3,  4,  5,  6,  7,  8, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50]
        self.skeleton_points = []   
        if not self.use_aria:
            self.json_list = self.get_jsons(self.sequences)
        else:
            self.json_list , self.aria_list = self.get_jsons(self.sequences)
        self.single_joint = opt['single_joint']
        self.video_model = opt['video_model']
        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.image_folder =  image_folder
        self.activities =  ["meal","cook", "recognition", "decoration", "work", "party", "clean", "golden"]
        #self.window_size = max(opt['window_size'], opt['cond_window_size']) 
        print('Dataset lenght: {}'.format(len(self.sequences)))
        print('Fold: {}'.format(self.fold))
    
    def get_activity(self, path):
        for activity in self.activities:
            if activity in path:
                return activity

    def get_jsons(self,sequences):
        json_list = []
        if self.use_aria:
            aria_list = []
        for sequence in sequences:
            if not self.use_aria:
                indv_json = json.load(open(os.path.join(self.root_sequences,sequence,"skeleton.json")))
                json_list.append(indv_json)
            else: 
                indv_json =  json.load(open(os.path.join(self.root_sequences,sequence,"skeleton.json")))
                json_list.append(indv_json)
                aria_json = json.load(open(os.path.join(self.root_sequences,sequence,"trajectory.json")))
                aria_list.append(aria_json)
        if self.use_aria:
            return json_list, aria_list
        else:
            return json_list

    def __getitem__(self, index):

        sequence = self.sequences[index]
        skeleton_sequence = self.json_list[index]
        if self.use_aria:
            aria_sequence = self.aria_list[index]
        capture_timestamps = sorted(list(skeleton_sequence.keys()))
        if self.future:
            timestamp_ns =  random.randint(self.slice_window, len(capture_timestamps)-self.future_frames)
        else:
            timestamp_ns =  random.randint(self.slice_window, len(capture_timestamps)-1)
        skeletons_window = []
        skeletons_window_future = []
        aria_window = []
        timestamps_window = capture_timestamps[timestamp_ns-self.slice_window:timestamp_ns]
        if self.future:
            timestamps_window_future = capture_timestamps[timestamp_ns:timestamp_ns+self.future_frames]
        #choose images
        if self.video_model:
            images_window = []
            image_paths =  os.path.join(self.image_folder,sequence, f"{timestamps_window[-1]}.npy")
            images_window_ = np.load(image_paths)
            images_window_ = images_window_[-self.slice_window:]

            for image in images_window_:
                if self.image_transforms is not None:
                    image =  self.image_transforms(image)
                images_window.append(image)
            images_window =  torch.stack(images_window)


        aria_window_future=[]
        for timestamp in timestamps_window_future:
            skeleton =  skeleton_sequence[timestamp]
            aria =  aria_sequence[timestamp]
            aria_window_future.append(aria["translation"] + aria["rotation"])
            skeletons_window_future.append(skeleton)
        for timestamp in timestamps_window:            
            skeleton =  skeleton_sequence[timestamp]
            if self.use_aria:
                aria =  aria_sequence[timestamp]
                aria_window.append(aria["translation"] + aria["rotation"]) # 0:3 is transltaion / 3:7 is rotation
            skeletons_window.append(skeleton)

        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        skeletons_window_future =  torch.Tensor(np.array(skeletons_window_future))
        if not self.use_aria:
            head_offset = (skeletons_window[:,4,:]).unsqueeze(1).repeat(1,21,1)
        else:
            aria_window =  torch.Tensor(np.array(aria_window))
            head_offset = (aria_window[:,0:3]).unsqueeze(1).repeat(1,21,1) # offset is using the translation of the aria device
            if self.future:
                aria_window_future =  torch.Tensor(np.array(aria_window_future))
                head_offset_future = (aria_window_future[:,0:3]).unsqueeze(1).repeat(1,21,1) # offset is using the translation of the aria device

                
        
        #skeletons_window_future = skeletons_window_future - head_offset_future
        #skeletons_window = skeletons_window - head_offset
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
            condition = skeletons_window[:,[4,8,12],:]
        
        activity = self.get_activity(sequence)
        if  self.video_model:
            return {'image':images_window,
                    'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window_future if self.output=='skeleton' else aria_window_future,
                    'offset':head_offset,
                    'activity': activity,
                    'aria_future': aria_window_future if self.future else None}
        else:
            return {'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window_future if self.output=='skeleton' else aria_window_future,
                    'offset':head_offset,
                    'activity': activity,
                    'aria_future': aria_window_future if self.future else None}
    
    def __len__(self):
        return len(self.sequences)

class Dataset_ADT_test(Dataset):
    def __init__(self, opt, root_sequences="/media/lambda001/SSD5/cdforigua/ADT/", fold=2, distribution_folder="./data", skeleton_occlusions=False, slice_window=None,
                image_transforms=None, image_folder="undistorted_images"):
        super(Dataset_ADT_test,self).__init__()
        self.root_sequences = root_sequences
        self.fold = opt['fold']
        self.use_aria = opt["use_aria"]
        self.use_rot = opt["use_rot"]
        self.skeleton_occlusions =skeleton_occlusions
        self.slice_window =  opt['window_size']+1
        self.future = opt["future"]
        self.output = opt["output"]
        self.opt = opt
        self.pred_input = opt['pred_input']
        # load sequences paths
        self.sequences = json.load(open(os.path.join(distribution_folder, "distribution_skeleton_filter.json")))["fold1" if self.fold==1 else "fold2"]
        self.sequences =  [sequence.split("/")[-1] for sequence in self.sequences]
        self.stream_id = StreamId("214-1") #rgb_image
        self.joint_labels = AriaDigitalTwinSkeletonProvider.get_joint_labels()
        self.joint_connections = AriaDigitalTwinSkeletonProvider.get_joint_connections()
        self.joint_idxs = [ 0 ,1,  2,  3,  4,  5,  6,  7,  8, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50]
        self.skeleton_points = [] 
        if not self.use_aria:
            self.json_list = self.get_jsons(self.sequences)
        else:
            self.json_list , self.aria_list = self.get_jsons(self.sequences)  
        self.single_joint = opt['single_joint']
        self.video_model = opt['video_model']
        self.image_transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.image_folder = image_folder
        self.activities =  ["meal","cook", "recognition", "decoration", "work", "party", "clean", "golden"]
        #self.window_size = max(opt['window_size'], opt['cond_window_size']) 
        print('Dataset lenght: {}'.format(len(self.sequences)))
        print('Fold: {}'.format(self.fold))

    def get_jsons(self,sequences):
        json_list = []
        if self.use_aria:
            aria_list = []
        for sequence in sequences:
            if not self.use_aria:
                indv_json = json.load(open(os.path.join(self.root_sequences,sequence,"skeleton.json")))
                json_list.append(indv_json)
            else: 
                indv_json =  json.load(open(os.path.join(self.root_sequences,sequence,"skeleton.json")))
                json_list.append(indv_json)
                aria_json = json.load(open(os.path.join(self.root_sequences,sequence,"trajectory.json")))
                aria_list.append(aria_json)
        if self.use_aria:
            return json_list, aria_list
        else:
            return json_list 

    def get_activity(self, path):
        for activity in self.activities:
            if activity in path:
                return activity
                
    def __getitem__(self, index):
        
        sequence = self.sequences[index]
        skeleton_sequence = self.json_list[index]
        if self.use_aria:
            aria_sequence = self.aria_list[index]
        capture_timestamps = sorted(list(skeleton_sequence.keys()))
        
        #choose images
        images_window = []
        skeletons_window = []
        aria_window = []
        for timestamp in capture_timestamps:
            if self.video_model:
                image_path =  os.path.join(self.root_sequences,sequence,self.image_folder, f"{timestamp}.png")
                image =  Image.open(image_path)
                if self.image_transforms is not None:
                    image =  self.image_transforms(image)
                images_window.append(image)
            skeleton =  skeleton_sequence[timestamp]
            skeletons_window.append(skeleton)
            if self.use_aria:
                aria =  aria_sequence[timestamp]
                aria_window.append(aria["translation"] + aria["rotation"])
        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        #images_window =  torch.Tensor(np.array(images_window, dtype=np.float64))
        if self.video_model:
            images_window =  torch.stack(images_window)
        if not self.use_aria:
            head_offset = (skeletons_window[:,4,:]).unsqueeze(1).repeat(1,21,1)
        else:
            aria_window =  torch.Tensor(np.array(aria_window))
            head_offset = (aria_window[:,0:3]).unsqueeze(1).repeat(1,21,1) # offset is using the translation of the aria device
        cond_window = skeletons_window
        cond_offset = head_offset

        if self.pred_input:
            container = np.load(os.path.join(self.opt['route'],sequence+'.npz'))
            pr_ = container['pr']
            pr_ = torch.from_numpy(pr_)
            cond_window = pr_
            #cond_offset = (torch.from_numpy(container['pred_aria'][:,:3])).unsqueeze(1).repeat(1,21,1)
        #cond_window = cond_window - cond_offset
        #skeletons_window = skeletons_window - head_offset
        # if self.future:
        #     aria_window = aria_window[:-1]

        if self.single_joint:
            if not self.use_aria:
                condition =  head_offset[:,0,:]
                skeletons_window[:,4,:]=skeletons_window[:,4,:]+head_offset[:,0,:]
                head_offset[:,4,:]=head_offset[:,4,:]-head_offset[:,4,:]
            else:
                if self.use_rot:
                    if self.output=='skeleton':
                        condition = torch.cat((cond_window.reshape(cond_window.shape[0],-1), aria_window),1)
                    else:
                        condition =  aria_window[:] # Takes both trans and rot
                else:
                    condition =  aria_window[:,0:3]
        else:
            condition = cond_window[:,[4,8,12],:]

        activity = self.get_activity(sequence)
        #breakpoint()
        if self.video_model:
            return {'image':images_window, 
                    'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window ,#if self.output=='skeleton' else aria_window,
                    'offset':head_offset,
                    'activity': activity,
                    'sequence':sequence,
                    'timestamps':capture_timestamps,
                    'cond_offset':cond_offset}
        else:
            return {'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window, #if self.output=='skeleton' else aria_window,
                    'offset':head_offset,
                    'activity': activity,
                    'sequence':sequence,
                    'timestamps':capture_timestamps,
                    'aria_future':condition,
                    'cond_offset':cond_offset}#FIX THIS, IT IS NOT REAL
        
    def __len__(self):
        return len(self.sequences)

if __name__=="__main__":
    dataset = Dataset_ADT_test({"fold":1, "single_joint":True, "window_size":40, "video_model":False, "use_aria":True, "use_rot":True},root_sequences="/media/lambda001/SSD5/cdforigua/ADT/", fold=1 ,distribution_folder=".")
    dataset.__getitem__(0)