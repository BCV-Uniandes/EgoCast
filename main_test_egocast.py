import os.path
import math
import argparse
import json
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_transform
import pickle
#from utils import utils_visualize as vis
from matplotlib import pyplot as plt
from tqdm import tqdm
save_animation = False
resolution = (800,800)
def save_videos(gt,pr,video_dir):
    # get all available skeletons in a sequence
    joint_idxs = [ 0 ,1,  2,  3,  4,  5,  6,  7,  8, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50]
    dict_joints = {k: v for v, k in enumerate(joint_idxs)}
    joint_connections = [(4, 3), (3, 2), (2, 1), (1, 0), (0, 43), (43, 44), (44, 45), (45, 46), (0, 47), (47, 48), (48, 49), (49, 50), (2, 5), (5, 6), (6, 7), (7, 8), (2, 24), (24, 25), (25, 26), (26, 27)]
    joint_labels = ['Skeleton', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand', 'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 'LIndex3', 'LMiddle1', 'LMiddle2', 'LMiddle3', 'LRing1', 'LRing2', 'LRing3', 'LPinky1', 'LPinky2', 'LPinky3', 'RShoulder', 'RUArm', 'RFArm', 'RHand', 'RThumb1', 'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1', 'RMiddle2', 'RMiddle3', 'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3', 'LThigh', 'LShin', 'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe']
    traces = []
    # draw skeleton
    if not os.path.exists(video_dir):
                os.makedirs(video_dir) 
    frames = min(900,len(gt))
    for idx in tqdm(range(0,frames)):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i in range(0, len(joint_connections)):
            gt_1 = gt[idx][dict_joints[joint_connections[i][0]]].cpu()
            gt_2 = gt[idx][dict_joints[joint_connections[i][1]]].cpu()
            pr_1 = pr[idx][dict_joints[joint_connections[i][0]]].cpu()
            pr_2 = pr[idx][dict_joints[joint_connections[i][1]]].cpu()
            ax.scatter([gt_1[0], gt_2[0]], [gt_1[1], gt_2[1]], [gt_1[2], gt_2[2]],alpha=0.5,c='red')
            ax.plot([gt_1[0], gt_2[0]], [gt_1[1], gt_2[1]], [gt_1[2], gt_2[2]],alpha=0.5,c='red')
            ax.scatter([pr_1[0], pr_2[0]], [pr_1[1], pr_2[1]], [pr_1[2], pr_2[2]],alpha=1,c='blue')
            ax.plot([pr_1[0], pr_2[0]], [pr_1[1], pr_2[1]], [pr_1[2], pr_2[2]],alpha=1,c='blue')
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

def main(json_path='options/test_avatarposer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['test']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'train':
            continue
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    model.init_test()

    pos_error = []
    vel_error = []
    
    inference_dict = {}
    gt_dict = {}

    for index, test_data in enumerate(test_loader):
        logger.info("testing the sample {}/{}".format(index, len(test_loader)))

        model.feed_data(test_data, test=True)

        #model.test_forecast(1)
        #model.test_forecast(3)
        #model.test_forecast(5)
        #model.test_forecast(15)
        
        model.test()
        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()
        #predicted_angle = body_parms_pred['pose_body']
        predicted_position = body_parms_pred['position']
        #predicted_body = body_parms_pred['body']
   
        #gt_angle = body_parms_gt['pose_body']
        gt_position = body_parms_gt['position']
        #gt_body = body_parms_gt['body']


        if index in [12, 52, 30, 36] and save_animation:
            video_dir = os.path.join(opt['path']['images'], str(index))
            save_videos(gt_position,predicted_position,video_dir)
        

        predicted_position = predicted_position#.cpu().numpy()
        gt_position = gt_position#.cpu().numpy()

        #predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)                    
        #gt_angle = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)

        data = model.visible[0]*torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))
        pos_error_ = data.sum()/(data!=0).sum()
        #pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])

        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*60
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60
        #vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))
        data_vel = model.visible[0]*torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))
        vel_error_  = data_vel.sum()/(data_vel!=0).sum()

        pos_error.append(pos_error_)
        vel_error.append(vel_error_)

        #pos_error_hands.append(pos_error_hands_)


        visible = model.visible.squeeze(0).unsqueeze(2).repeat(1,1,3)
        visible[visible!=1]=torch.nan

        #t_ = test_data['t'][0]
        t_ = test_data['t']
        gt_nan = visible*gt_position
        #preds_ = dict(zip(t_,predicted_position.tolist()))
        #gt_ = dict(zip(t_,gt_nan.tolist()))

        preds_ = dict(zip([str(x[0]) for x in t_], predicted_position.tolist()))
        gt_ = dict(zip([str(x[0]) for x in t_], gt_nan.tolist()))
     
        inference_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":preds_}
        gt_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":gt_}

    pred_path = os.path.join(opt['path']['images'],dataset_opt['split']+f'_pred.json')
    gt_path = os.path.join(opt['path']['images'],dataset_opt['split']+f'_gt.json')
    

    with open(pred_path, 'w') as fp:
        json.dump(inference_dict, fp)
    with open(gt_path, 'w') as fp:
        json.dump(gt_dict, fp)

    pos_error = [t for t in pos_error if not torch.isnan(t)]
    vel_error = [t for t in vel_error if not torch.isnan(t)]

    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)
    #pos_error_hands = sum(pos_error_hands)/len(pos_error_hands)


    # testing log
    logger.info('Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}\n'.format(pos_error*100, vel_error*100))


    

if __name__ == '__main__':
    main()
