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
from mpl_toolkits.mplot3d import Axes3D
import json



def wandb_3dplot(gt,pr,visibility,video_dir,pos,vel,aria):
    if not os.path.exists(video_dir):
                os.makedirs(video_dir) 

    # Create and log the evolving 3D pose plot
    joint_connections = [(1,4),(1,2),(2,12),(12,7),(2,5),(2,15),(15,3),(3,10),(5,13),(13,0),(5,9),(9,8),(8,14),(9,6),(6,16),(16,11),(6,15),(5,15)]
    for t, (keypoints,pr,visibility,aria_) in enumerate(tqdm(zip(gt,pr,visibility,aria[0]))):
        # Convert the PyTorch tensor to a NumPy array for plotting
        keypoints_np = keypoints.numpy()
        pr_np = pr.numpy()
        aria_np = aria_.numpy()

        # Create a Matplotlib 3D plot for each time step
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract x, y, and z coordinates from keypoints
        x = keypoints_np[:, 0]
        y = keypoints_np[:, 1]
        z = keypoints_np[:, 2]

        # Extract x, y, and z coordinates from keypoints
        x_pr = pr_np[:, 0]
        y_pr = pr_np[:, 1]
        z_pr = pr_np[:, 2]

        # Extract x, y, and z coordinates from keypoints
        x_aria = aria_np[0]
        y_aria = aria_np[1]
        z_aria = aria_np[2]

        ax.scatter(x_aria, y_aria, z_aria, c='r', marker='o')

        # Plot the keypoints with different colors based on visibility
        for i in range(len(x)):
            if visibility[i] != 0:
                ax.scatter(x[i], y[i], z[i], c='b', marker='o')

        for connection in joint_connections:
            joint1, joint2 = connection
            if visibility[joint1] == 1 and visibility[joint2] == 1:
                ax.plot([x[joint1], x[joint2]], [y[joint1], y[joint2]], [z[joint1], z[joint2]], c='b')

        # Plot the keypoints with different colors based on visibility
        for i in range(len(x)):
            if visibility[i] != 0:
                ax.scatter(x_pr[i], y_pr[i], z_pr[i], c='g', marker='o')

        for connection in joint_connections:
            joint1, joint2 = connection
            if visibility[joint1] == 1 and visibility[joint2] == 1:
                ax.plot([x_pr[joint1], x_pr[joint2]], [y_pr[joint1], y_pr[joint2]], [z_pr[joint1], z_pr[joint2]], c='g')

        err = np.around(pos.cpu().numpy(),2)
        ve = np.around(vel.cpu().numpy(),2)
        ax.set_title('Sequence:'+video_dir.split('/')[-1]+' MPJPE:'+str(err)+ ' MPJVE:'+str(ve))
        ax.view_init(elev=111, azim=0)

        
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.savefig(os.path.join(video_dir,str(t).zfill(5)+'.png'))
        plt.close()


def save_videos(gt,pr,visibility,video_dir):

    joint_connections = [(1,4),(1,2),(2,12),(12,7),(2,5),(2,15),(15,3),(3,10),(5,13),(13,0),(5,9),(9,8),(8,14),(9,6),(6,16),(16,11),(6,15),(5,15)]

    # draw skeleton
    if not os.path.exists(video_dir):
                os.makedirs(video_dir) 
    frames = len(gt)#min(900,len(gt))
    for idx in tqdm(range(0,frames)):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i in range(0, len(joint_connections)):
            if visibility[idx][joint_connections[i][0]] == 0 or visibility[idx][joint_connections[i][1]] == 0:
                continue
            gt_1 = gt[idx][joint_connections[i][0]].cpu()
            gt_2 = gt[idx][joint_connections[i][1]].cpu()
            pr_1 = pr[idx][joint_connections[i][0]].cpu()
            pr_2 = pr[idx][joint_connections[i][1]].cpu()
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

def main(json_path='options/train_egocast_imu.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    wandb.init(project="EgoCast",config=opt, mode = opt['wandb_mode'], name=opt['wandb_name'])
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


    for phase, dataset_opt in opt['datasets'].items():
   
        if phase == 'train':
     
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
                                     shuffle=False, num_workers=1,
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
    constant = np.load('data/mean_with_off.npy')
    constant = torch.from_numpy(constant)
    test_step = 0
    test_first = False
    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            
            current_step += 1
            if not test_first:
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
                pos_error_hands = []
                pos_error_baseline = []
                tasks = []
                test_step+=1
                inference_dict = {}
                gt_dict = {}
                baseline_dict = {}
                
                for index, test_data in enumerate(test_loader):

                    logger.info("testing the sample {}/{}".format(index, len(test_loader)))
                    model.feed_data(test_data, test=True)
                    #tasks.append(test_data['task'])
                    
                    model.test()

                    body_parms_pred = model.current_prediction()
                    body_parms_gt = model.current_gt()
                    #predicted_angle = body_parms_pred['pose_body']
                    predicted_position = body_parms_pred['position']
                    #predicted_body = body_parms_pred['body']

                    #gt_angle = body_parms_gt['pose_body']
                    gt_position = body_parms_gt['position']
                    #gt_body = body_parms_gt['body']


                    predicted_position = predicted_position#.cpu().numpy()
                    gt_position = gt_position#.cpu().numpy()

                    constant_train = constant.repeat(gt_position.shape[0],1,1).cuda()
                    if opt['datasets']['test']['use_rot']:
                        constant_train = constant_train + test_data['offset'][0][:,:,:3].cuda()
                    else:
                        constant_train = constant_train + test_data['offset'][0].cuda()
                    data_baseline = model.visible[0]*torch.sqrt(torch.sum(torch.square(gt_position-constant_train),axis=-1))
                    pos_error_baseline_ = data_baseline.sum()/(data_baseline!=0).sum()                    

                    data = model.visible[0]*torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))
                    pos_error_ = data.sum()/(data!=0).sum()

                    gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*10
                    predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*10

                    data_vel = model.visible[0]*torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))
                    vel_error_  = data_vel.sum()/(data_vel!=0).sum()

                    if save_animation: #index in [12, 52, 30, 36] and 
                        video_dir = os.path.join(opt['path']['images'], str(index))
                        wandb_3dplot(test_data['gt'][0].cpu(),predicted_position.cpu(),test_data['visible'][0],video_dir,pos_error_*100,vel_error_*100,test_data['cond'])
                        #save_videos(gt_position,predicted_position,model.visible[0],video_dir)
                    #test_data['gt'][0].cpu()gt_position.cpu()
                    if model.visible.max() !=0:
                        pos_error.append(pos_error_)
                        pos_error_baseline.append(pos_error_baseline_)
                        vel_error.append(vel_error_)
                        tasks.append(str(test_data['task'].numpy()[0])[0])

                    visible = model.visible.squeeze(0).unsqueeze(2).repeat(1,1,3)
                    visible[visible!=1]=torch.nan
                    gt_nan = visible*gt_position

                    t_ = np.array(test_data['t']).squeeze(1).tolist()
                    preds_ = dict(zip(t_,predicted_position.tolist()))
                    baseline_ = dict(zip(t_,constant_train.tolist()))
                    gt_ = dict(zip(t_,gt_nan.tolist()))
                    inference_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":preds_}
                    gt_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":gt_}
                    baseline_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":baseline_}
                pred_path = os.path.join(opt['path']['images'],dataset_opt['split']+'_pred.json')
                gt_path = os.path.join(opt['path']['images'],dataset_opt['split']+'_gt.json')
                baseline_path = os.path.join(opt['path']['images'],dataset_opt['split']+'_baseline.json')
                with open(pred_path, 'w') as fp:
                    json.dump(inference_dict, fp)
                with open(gt_path, 'w') as fp:
                    json.dump(gt_dict, fp)
                with open(baseline_path, 'w') as fp:
                    json.dump(baseline_dict, fp)

                activities ={'1':'Cooking','2':'Health','3':'Campsite','4':'Bike repair','5':'Music','6':'Basketball','7':'Bouldering','8':'Soccer','9':'Dance'}
                for task_num in range(0,10):
                    task_ids = [i for i, j in enumerate(tasks) if j == str(task_num)]
                    if len(task_ids)>0:
                        pos_ = torch.stack(pos_error)[task_ids].cpu().numpy()
                        vel_ = torch.stack(vel_error)[task_ids].cpu().numpy()
                        pos_b = torch.stack(pos_error_baseline)[task_ids].cpu().numpy()
                        logger.info('<epoch:{:3d}, iter:{:8,d}, Task: {}, Samples: {}, MPJPE[cm]: {:<.5f}, baseline MPJPE[cm]: {:<.5f}, MPJVE [cm/s]: {:<.5f}\n'.format(epoch, current_step,activities[str(task_num)],len(task_ids), (pos_.mean())*100, (pos_b.mean())*100, (vel_.mean())*100))
                        

                
                pos_error = sum(pos_error)/len(pos_error)
                pos_error_baseline = sum(pos_error_baseline)/len(pos_error_baseline)
                vel_error = sum(vel_error)/len(vel_error)
                #pos_error_hands = sum(pos_error_hands)/len(pos_error_hands)

                wandb.log({'MPJPE':pos_error*100,'MPJVE':vel_error*100,'test_step':test_step})
                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Positional error baseline [cm/s]: {:<.5f}\n'.format(epoch, current_step,pos_error*100, vel_error*100,pos_error_baseline*100))


    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
