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
        self.slice_window =  opt['window_size']+1
        self.use_aria = opt["use_aria"]
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
        timestamp_ns =  random.randint(self.slice_window, len(capture_timestamps)-1)
        skeletons_window = []
        aria_window = []
        timestamps_window = capture_timestamps[timestamp_ns-self.slice_window: timestamp_ns]
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

        
        for timestamp in timestamps_window:            
            skeleton =  skeleton_sequence[timestamp]
            if self.use_aria:
                aria =  aria_sequence[timestamp]
                aria_window.append(aria["translation"] + aria["rotation"]) # 0:3 is transltaion / 3:7 is rotation
            skeletons_window.append(skeleton)

        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        if not self.use_aria:
            head_offset = (skeletons_window[:,4,:]).unsqueeze(1).repeat(1,21,1)
        else:
            aria_window =  torch.Tensor(np.array(aria_window))
            head_offset = (aria_window[:,0:3]).unsqueeze(1).repeat(1,21,1) # offset is using the translation of the aria device

        skeletons_window = skeletons_window - head_offset
        if self.single_joint:
            if not self.use_aria:
                condition =  head_offset[:,0,:]
                skeletons_window[:,4,:]=skeletons_window[:,4,:]+head_offset[:,0,:]
                head_offset[:,4,:]=head_offset[:,4,:]-head_offset[:,4,:]
            else:
                condition =  aria_window[:,0:3] # from 0:3 to just translation vector
                skeletons_window = torch.cat((skeletons_window, aria_window[:,:3].unsqueeze(1)),1) # add the translation vector to the skeleton
        else:
            condition = skeletons_window[:,[4,8,12],:]

        activity = self.get_activity(sequence)
        if  self.video_model:
            return {'image':images_window,
                    'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window,
                    'offset':head_offset,
                    'activity': activity}
        else:
            return {'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window,
                    'offset':head_offset,
                    'activity': activity}
    
    def __len__(self):
        return len(self.sequences)

class Dataset_ADT(Dataset):
    def __init__(self, opt, root_sequences="/media/lambda001/SSD5/cdforigua/ADT/", fold=1, distribution_folder="./data", skeleton_occlusions=False, slice_window=41,
                image_transforms=None, image_folder="undistorted_images"):
        super(Dataset_ADT,self).__init__()
        self.root_sequences = root_sequences
        self.fold = opt['fold']
        self.skeleton_occlusions =skeleton_occlusions
        self.slice_window =  slice_window
        # load sequences paths
        self.sequences = json.load(open(os.path.join(distribution_folder, "distribution_skeleton.json")))["fold1" if self.fold==1 else "fold2"]
        self.sequences =  [sequence.split("/")[-1] for sequence in self.sequences]
        self.stream_id = StreamId("214-1") #rgb_image
        self.joint_labels = AriaDigitalTwinSkeletonProvider.get_joint_labels()
        self.joint_connections = AriaDigitalTwinSkeletonProvider.get_joint_connections()
        self.joint_idxs = [ 0 ,1,  2,  3,  4,  5,  6,  7,  8, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50]
        self.skeleton_points = []   
        self.json_list = self.get_jsons(self.sequences)
        self.single_joint = opt['single_joint']
        self.image_transforms = T.Compose([
            T.Resize(224),
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
        for sequence in sequences:
            indv_json = json.load(open(os.path.join(self.root_sequences,sequence,"skeleton.json")))
            json_list.append(indv_json)
        return json_list

    def __getitem__(self, index):
        sequence = self.sequences[index]
        skeleton_sequence = self.json_list[index]
        capture_timestamps = sorted(list(skeleton_sequence.keys()))
        timestamp_ns =  random.randint(self.slice_window, len(capture_timestamps)-self.slice_window)
        #choose images
        images_window = []
        skeletons_window = []
        
        timestamps_window = capture_timestamps[timestamp_ns-self.slice_window: timestamp_ns]

        for timestamp in timestamps_window:
            image_path =  os.path.join(self.root_sequences,sequence,self.image_folder, f"{timestamp}.png")
            image =  Image.open(image_path)
            if self.image_transforms is not None:
                image =  self.image_transforms(image)
            images_window.append(image)
            skeleton =  skeleton_sequence[timestamp]
            skeletons_window.append(skeleton)
            
        skeletons_window =  torch.Tensor(np.array(skeletons_window))
        images_window =  torch.stack(images_window)
        
        head_offset = (skeletons_window[:,4,:]).unsqueeze(1).repeat(1,21,1)
        
        skeletons_window = skeletons_window - head_offset
        if self.single_joint:
            condition =  head_offset[:,0,:]
            skeletons_window[:,4,:]=skeletons_window[:,4,:]+head_offset[:,0,:]
            head_offset[:,4,:]=head_offset[:,4,:]-head_offset[:,4,:]
        else:
            condition = skeletons_window[:,[4,8,12],:]
        
        activity = self.get_activity(sequence)
        return {'image':images_window,
                'cond': condition, #IDs for each part of the body
                'gt': skeletons_window,
                'offset':head_offset,
                'activity': activity}
    
    def __len__(self):
        return len(self.sequences)


class Dataset_ADT_test(Dataset):
    def __init__(self, opt, root_sequences="/media/lambda001/SSD5/cdforigua/ADT/", fold=2, distribution_folder="./data", skeleton_occlusions=False, slice_window=None,
                image_transforms=None, image_folder="undistorted_images"):
        super(Dataset_ADT_test,self).__init__()
        self.root_sequences = root_sequences
        self.fold = opt['fold']
        self.use_aria = opt["use_aria"]
        self.skeleton_occlusions =skeleton_occlusions
        self.slice_window =  opt['window_size']+1
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
        skeletons_window = skeletons_window - head_offset
        if self.single_joint:
            if not self.use_aria:
                condition =  head_offset[:,0,:]
                skeletons_window[:,4,:]=skeletons_window[:,4,:]+head_offset[:,0,:]
                head_offset[:,4,:]=head_offset[:,4,:]-head_offset[:,4,:]
            else:
                condition =  aria_window[:,0:3]
        else:
            condition = skeletons_window[:,[4,8,12],:]

        activity = self.get_activity(sequence)

        if self.video_model:
            return {'image':images_window, 
                    'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window,
                    'offset':head_offset,
                    'activity': activity}
        else:
            return {'cond': condition, #IDs for each part of the body
                    'gt': skeletons_window,
                    'offset':head_offset,
                    'activity': activity}
        
    def __len__(self):
        return len(self.sequences)

if __name__=="__main__":
    dataset = Dataset_ADT_test({"fold":1, "single_joint":True, "window_size":40, "video_model":False, "use_aria":True},root_sequences="/media/lambda001/SSD5/cdforigua/ADT/", fold=1 ,distribution_folder=".")
    dataset.__getitem__(0)