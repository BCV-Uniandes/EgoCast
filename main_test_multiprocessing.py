import multiprocessing
import os.path
import math
import argparse
import random
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from data.dataset_egoexo import my_collate_multiprocessing
from models.select_model import define_Model
from utils import utils_transform
import pickle
#from utils import utils_visualize as vis
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
save_animation = True
resolution = (800,800)


def parallel_evaluation(num_processes=5, json_path='options/test_egoexo.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    
    json_path = parser.parse_args().opt
    print(json_path)
    # Assuming a fixed number of samples for simplicity. 
    # In a real-world scenario, this would be determined dynamically.
    
    total_samples = 442
    samples_per_process = total_samples // num_processes
    # Create argument tuples for each process
    custom_indices = [1,2,3,5,7]
    args = [(i, i + samples_per_process, json_path, idx) for i, idx in zip(range(0, total_samples, samples_per_process), custom_indices)]
    # Set CUDA_VISIBLE_DEVICES for each process
    for idx in [1,2,3,5,7]:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(wrapper_function, args)

    # Consolidate the results
    pos_errors = [result[0] for result in results]
    vel_errors = [result[1] for result in results]

    avg_pos_error = sum(pos_errors) / len(pos_errors)
    avg_vel_error = sum(vel_errors) / len(vel_errors)

    return avg_pos_error, avg_vel_error
def wrapper_function(args):
    return modified_main(*args)

def modified_main(start_idx, end_idx, json_path='options/test_egoexo.json', idx_gpu=1):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    save_interval = 30


    opt = option.parse(parser.parse_args().opt, is_train=True)
    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

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
            if dataset_opt["video_model"]:
                test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                        shuffle=False, num_workers=0,
                                        drop_last=False, pin_memory=True, collate_fn=my_collate_multiprocessing)
            else:
                test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                        shuffle=False, num_workers=0,
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

        if index < start_idx or index >= end_idx:  # Skip data outside the specified range
            continue
        logger.info("testing the sample {}/{}".format(index, len(test_loader)))
        model.feed_data(test_data, test=True)

        try:

            model.test(image_transforms=test_set.image_transforms)

        except:
            continue
       
        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()

        predicted_position = body_parms_pred['position']
        
        gt_position = body_parms_gt['position']
        
        data = model.visible[0]*torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))
        pos_error_ = data.sum()/(data!=0).sum()
       
        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*10
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*10

        data_vel = model.visible[0]*torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))
        vel_error_  = data_vel.sum()/(data_vel!=0).sum()

        if model.visible.max() !=0:
            pos_error.append(pos_error_.cpu().numpy())
            vel_error.append(vel_error_.cpu().numpy())
       
            print(pos_error_, vel_error_)

        visible = model.visible.squeeze(0).unsqueeze(2).repeat(1,1,3)
        visible[visible!=1]=torch.nan
        gt_nan = visible*gt_position
        t_ = test_data['t'][0]
        preds_ = dict(zip(t_,predicted_position.tolist()))
 
        gt_ = dict(zip(t_,gt_nan.tolist()))
        inference_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":preds_}
        gt_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":gt_}

        if (index + 1) % save_interval == 0:
            intermediate_pred_path = os.path.join(opt['path']['images'], f"{dataset_opt['split']}_{str(idx_gpu)}_pred_{index + 1}.json")
            intermediate_gt_path = os.path.join(opt['path']['images'], f"{dataset_opt['split']}_{str(idx_gpu)}_gt_{index + 1}.json")
            with open(intermediate_pred_path, 'w') as fp:
                json.dump(inference_dict, fp)
            with open(intermediate_gt_path, 'w') as fp:
                json.dump(gt_dict, fp)

  
    pred_path = os.path.join(opt['path']['images'],dataset_opt['split']+f'_{str(idx_gpu)}_pred.json')
    gt_path = os.path.join(opt['path']['images'],dataset_opt['split']+f'_{str(idx_gpu)}_gt.json')

    with open(pred_path, 'w') as fp:
        json.dump(inference_dict, fp)
    with open(gt_path, 'w') as fp:
        json.dump(gt_dict, fp)

    
    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)
        
    return pos_error, vel_error

if __name__ == '__main__':
    #avg_pos_error, avg_vel_error = modified_main(0,244)
    avg_pos_error, avg_vel_error = parallel_evaluation(num_processes=5)
    print("-*-"*10)
    print("final avg_pos_error: ", avg_pos_error*100)
    print("final avg_vel_error: ", avg_vel_error*100)