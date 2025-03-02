import os.path
import math
import wandb
import argparse
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
import torchvision.transforms as T
import pickle
#from utils import utils_visualize as vis
from matplotlib import pyplot as plt
from tqdm import tqdm

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


save_animation = False
resolution = (800,800)

def main(json_path='options/train_egocast_forecasting.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    wandb.init(project=opt['wandb_name'],config=opt, mode = opt['wandb_mode'])#name=opt['wandb_name'] mode="disabled"
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
    if init_path_G is not None:
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
    logger.info(option.dict2str(opt))
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    dataset_type = opt['datasets']['train']['dataset_type']

    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'train':
            breakpoint()
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True
                                      )
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=0,
                                     drop_last=False, pin_memory=True
                                    )
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

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    test_step = 0
    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):
            #breakpoint()
            current_step += 1
            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------

            model.feed_data(train_data)

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)
            wandb_dict = model.log_dict
            wandb_dict['train_step']=current_step
            wandb.log(wandb_dict)

            # -------------------------------
            # merge bnorm
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                pos_error = []
                vel_error = []
                aria_error = []
                rot_error = []
                pos_error_hands = []
                test_step+=1
                for index, test_data in enumerate(test_loader):
                    if index in [17,62]:
                        continue
                    logger.info("testing the sample {}/{}".format(index, len(test_loader)))

                    model.feed_data(test_data, test=True)

                    if opt['datasets']['test']['future']:  
                        if opt['datasets']['test']['output']=='aria':
                            pos_error_,rot_error_,vel_error_,_,_,_ = model.test_fcast()
                        else:
                            pos_error_,vel_error_,aria_error_,_,_,_ = model.test_fcast()
                    else:
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


                        pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
                        #pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])

                        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*30
                        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*30
                        vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

                    pos_error.append(pos_error_)
                    vel_error.append(vel_error_)
                    if opt['datasets']['test']['output']=='aria':
                        rot_error.append(rot_error_)
                    else:
                        aria_error.append(aria_error_)


                    #pos_error_hands.append(pos_error_hands_)



                pos_error = sum(pos_error)/len(pos_error)
                vel_error = sum(vel_error)/len(vel_error)

                if opt['datasets']['test']['output']=='aria':
                    rot_error = sum(rot_error)/len(rot_error)
                else:
                    aria_error = sum(aria_error)/len(aria_error)
                #pos_error_hands = sum(pos_error_hands)/len(pos_error_hands)
                if opt['datasets']['test']['output']=='aria':
                    wandb.log({'MPJPE':pos_error*100,'MPJVE':vel_error*100,'test_step':test_step,'MPJRE':rot_error})
                
                wandb.log({'MPJPE':pos_error*100,'MPJVE':vel_error*100,'aria':aria_error*100,'test_step':test_step})
                # testing log
                if opt['datasets']['test']['output']!='aria':
                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Average aria error [cm]: {:<.5f}\n'.format(epoch, current_step,pos_error*100, vel_error*100, aria_error*100))
                else:
                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average positional error [cm]: {:<.5f}, Average rotational error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}\n'.format(epoch, current_step,pos_error*100, rot_error, vel_error*100))



    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
