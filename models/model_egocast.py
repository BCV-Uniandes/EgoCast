from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from PIL import Image
from models.select_model import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss

from utils.utils_regularizers import regularizer_orth, regularizer_clip
#from human_body_prior.tools.angle_continuous_repres import geodesic_loss_R
from IPython import embed
from utils.utils_transform import bgs
from utils import utils_transform
from models.video_transformer import SpaceTimeTransformer
#from utils import utils_visualize as vis
#from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa

from tqdm import tqdm
from PIL import Image
import numpy as np 
import torchvision.transforms as T



class ModelEgoCast(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelEgoCast, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        self.window_size = self.opt['netG']['window_size']
        
        #self.video_transformer = SpaceTimeTransformer(num_frames=opt['datasets']['train']['window_size'])
        #self.bm = self.netG.module.body_model


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def init_test(self):
        self.load(test=True)                           # load model
        self.log_dict = OrderedDict()         # log
    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self, test=False):
        if self.opt['netG']['video_model']:
            load_path_video = 'model_zoo/egovlp.pth'
            print('Loading model for video [{:s}] ...'.format(load_path_video))
            model_video = torch.load(load_path_video)
            m_video_statedict = model_video['state_dict']
            #new_state_dict = {}
            #for (key, param) in m_video_statedict.items():
            #    if 'module.video_model' in key:
            #        new_key = key[19:]
            #        new_state_dict[new_key]=param
            dims = self.netG.state_dict()['module.video_model.temporal_embed'].shape
            new_temporal_embed = torch.zeros([m_video_statedict['module.video_model.temporal_embed'].shape[0], dims[1], dims[2]])
            new_temporal_embed[:, :m_video_statedict['module.video_model.temporal_embed'].shape[1]] = m_video_statedict['module.video_model.temporal_embed']
            m_video_statedict['module.video_model.temporal_embed'] = new_temporal_embed
            
            self.netG.load_state_dict(m_video_statedict, strict=False)
            
        load_path_G = self.opt['path']['pretrained_netG'] if test == False else self.opt['path']['pretrained']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict= not self.opt['netG']['video_model'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()
        



    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss(reduction='none').to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'geodesic':
            self.G_lossfn = geodesic_loss_R(reduction='mean')
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        self.G_lossaria = nn.L1Loss().to(self.device)

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)



    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True, test=False):



        #self.opt['netG']['video_model']:
        self.L = data['cond'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.offset = data['offset'].to(self.device)
        if self.opt['netG']['video_model']:
            self.image = data['image'].to(self.device)
        self.visible = data['visible'].to(self.device)
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        if self.opt['netG']['video_model']:
            self.E_global_orientation = self.netG(self.L,self.image)
            #self.E_global_orientation = self.netG(self.image)
        else:
            self.E_global_orientation = self.netG(self.L)


    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        
        self.G_optimizer.zero_grad()
        self.netG_forward()
        
        if not self.opt['netG']['use_rot']:
            global_orientation_loss = (self.visible[:,-1]*self.G_lossfn(self.E_global_orientation.reshape(self.gt.shape[0],17,3), self.gt[:,-1]).mean(axis=2))
            global_orientation_loss = global_orientation_loss.sum()/(global_orientation_loss!=0).sum()
            loss =  global_orientation_loss 
        else:
            self.E_skel = self.E_global_orientation[:,:51].reshape(self.gt.shape[0],17,3)
            self.aria_pos = self.E_global_orientation[:,51:54]
            self.aria_rot = self.E_global_orientation[:,54:]
            global_orientation_loss = (self.visible[:,-1]*self.G_lossfn(self.E_skel, self.gt[:,-1]).mean(axis=2))
            global_orientation_loss = global_orientation_loss.sum()/(global_orientation_loss!=0).sum()
            aria_pos_loss = self.G_lossaria(self.aria_pos,self.L[:,-1,:3])
            aria_rot_loss = self.G_lossaria(self.aria_rot,self.L[:,-1,3:])
            loss =  10*global_orientation_loss+aria_pos_loss+aria_rot_loss
            

        loss.backward()


        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['total_loss'] = loss.item()
        self.log_dict['global_orientation_loss'] = global_orientation_loss.item()
        if self.opt['netG']['use_rot']:
            self.log_dict['aria_pos_loss'] = aria_pos_loss.item()
            self.log_dict['aria_rot_loss'] = aria_rot_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
            
            self.netG.eval()

            self.L = self.L.squeeze()
            self.L = self.L.reshape(self.L.shape[0],-1)
            self.gt = self.gt.squeeze()
            self.offset = self.offset.squeeze()
            

            window_size = self.opt['datasets']['test']['window_size']
            self.forecast_sec = self.opt['datasets']['test']['forecast_sec']
            use_aria = self.opt['datasets']['test']['use_aria']
            use_rot= self.opt['datasets']['test']['use_rot']

            input_singleframe = False
            with torch.no_grad():

                if self.L.shape[0] < window_size:


                    if input_singleframe:
                        input_list = []
                        for frame_idx in range(0,self.L.shape[0]):
                            input_list.append(self.L[[frame_idx]].unsqueeze(0))
                        input_tensor = torch.cat(input_list, dim = 0)

                        E_global_orientation_tensor, E_joint_rotation_tensor = self.netG(input_tensor)
                    else:
                        E_global_orientation_list = []                     

                        for frame_idx in range(0,self.L.shape[0]):
                            E_global_orientation = self.netG(self.L[0:frame_idx+1].unsqueeze(0))
                            E_global_orientation_list.append(E_global_orientation)    
                        E_global_orientation_tensor = torch.cat(E_global_orientation_list, dim=0)

                else:  


                    input_list_1 = []
                    input_list_2 = []
                    if input_singleframe:
                        for frame_idx in range(0,window_size):
                            input_list_1.append(self.L[[frame_idx]].unsqueeze(0))
                        input_tensor_1 = torch.cat(input_list_1, dim = 0)
                        E_global_orientation_1, E_joint_rotation_1 = self.netG(input_tensor_1)
                    else:
                        E_global_orientation_list_1 = []        
                        for frame_idx in range(0,window_size):
                            E_global_orientation = self.netG(self.L[0:frame_idx+1].unsqueeze(0))
                            E_global_orientation_list_1.append(E_global_orientation)
                        E_global_orientation_1 = torch.cat(E_global_orientation_list_1, dim=0)

                    for frame_idx in range(window_size,self.L.shape[0]):
                        input_list_2.append(self.L[frame_idx-window_size:frame_idx+1,...].unsqueeze(0))
                    input_tensor_2 = torch.cat(input_list_2, dim = 0)
                    E_global_orientation_2 = self.netG(input_tensor_2)
 
                    E_global_orientation_tensor = torch.cat([E_global_orientation_1,E_global_orientation_2], dim=0)
            self.E_global_orientation = E_global_orientation_tensor
            self.E = E_global_orientation_tensor.to(self.device)
            if use_rot:
                self.E = self.E[:,:51]
            self.predicted_position = self.E.reshape(-1,17,3)
            self.gt_position = self.gt 
            self.predicted_position = self.predicted_position 

            self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_prediction(self,):
        body_parms = OrderedDict()
        body_parms['position'] = self.predicted_position
        return body_parms

    def current_gt(self, ):
        body_parms = OrderedDict()
        body_parms['position'] = self.gt_position
        return body_parms


    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
    



class ModelEgoCastEval(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelEgoCastEval, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        self.window_size = self.opt['netG']['window_size']
        
        #self.video_transformer = SpaceTimeTransformer(num_frames=opt['datasets']['train']['window_size'])
        #self.bm = self.netG.module.body_model


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def init_test(self):
        self.load(test=True)                           # load model
        self.log_dict = OrderedDict()         # log
    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self, test=False):
        if self.opt['netG']['video_model']:
            load_path_video = 'model_zoo/egovlp.pth'
            print('Loading model for video [{:s}] ...'.format(load_path_video))
            model_video = torch.load(load_path_video)
            m_video_statedict = model_video['state_dict']
            #new_state_dict = {}
            #for (key, param) in m_video_statedict.items():
            #    if 'module.video_model' in key:
            #        new_key = key[19:]
            #        new_state_dict[new_key]=param
            dims = self.netG.state_dict()['module.video_model.temporal_embed'].shape
            new_temporal_embed = torch.zeros([m_video_statedict['module.video_model.temporal_embed'].shape[0], dims[1], dims[2]])
            new_temporal_embed[:, :m_video_statedict['module.video_model.temporal_embed'].shape[1]] = m_video_statedict['module.video_model.temporal_embed']
            m_video_statedict['module.video_model.temporal_embed'] = new_temporal_embed
            
            self.netG.load_state_dict(m_video_statedict, strict=False)
            
        load_path_G = self.opt['path']['pretrained_netG'] if test == False else self.opt['path']['pretrained']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict= not self.opt['netG']['video_model'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()
        



    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss(reduction='none').to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'geodesic':
            self.G_lossfn = geodesic_loss_R(reduction='mean')
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)



    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True, test=False):



        #self.opt['netG']['video_model']:
        self.L = data['cond'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.offset = data['offset'].to(self.device)
        if self.opt['netG']['video_model']:
            self.image = data['image']
        self.visible = data['visible'].to(self.device)
        
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        if self.opt['netG']['video_model']:
            self.E_global_orientation = self.netG(self.L,self.image)
            #self.E_global_orientation = self.netG(self.image)
        else:
            self.E_global_orientation = self.netG(self.L)


    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        
        self.G_optimizer.zero_grad()
        self.netG_forward()
        global_orientation_loss = (self.visible[:,-1]*self.G_lossfn(self.E_global_orientation.reshape(24,17,3), self.gt[:,-1]).mean(axis=2))#self.G_lossfn(self.E_global_orientation, self.gt[:,-1].reshape(self.gt.shape[0],-1))
        global_orientation_loss = global_orientation_loss.sum()/(global_orientation_loss!=0).sum()
        #joint_rotation_loss = self.G_lossfn(self.E_joint_rotation, self.H_joint_rotation)
        #joint_position_loss = self.G_lossfn(self.E_joint_position, self.H_joint_position) 
        loss =  global_orientation_loss #0.02*global_orientation_loss #+ joint_rotation_loss + joint_position_loss
        loss.backward()


        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['total_loss'] = loss.item()
        self.log_dict['global_orientation_loss'] = global_orientation_loss.item()
        #self.log_dict['joint_rotation_loss'] = joint_rotation_loss.item()
        #self.log_dict['joint_position_loss'] = joint_position_loss.item()


        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def feed_cola(self, cola, frame_idx, image_transforms, window_size):
        image = Image.open(self.image[0][frame_idx])
        if image_transforms is not None:
            image =  image_transforms(image)
        if len(cola)<window_size+1: 
            cola.append(image)
        else:
            cola.pop(0)
            cola.append(image)
        return torch.stack(cola, dim=0)

    def feed_cola_preload(self, frame_idx, image_transforms):
        cola = []
        image_ = self.image_preload[0][frame_idx]
        image_ =  np.load(image_)
        for image in image_:
            if image_transforms is not None:
                image =  image_transforms(image)
            cola.append(image)
        return torch.stack(cola, dim=0)
    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self,image_transforms = None):
            cola = []
            self.netG.eval()
            
            self.L = self.L.squeeze()
            self.L = self.L.reshape(self.L.shape[0],-1)
            self.gt = self.gt.squeeze()
            #self.gt = self.gt.reshape(self.gt.shape[0],-1)
            self.offset = self.offset.squeeze()
            #self.offset = self.offset.reshape(self.offset.shape[0],-1)

            window_size = self.opt['datasets']['test']['window_size']
            self.forecast_sec = self.opt['datasets']['test']['forecast_sec']
            use_aria = self.opt['datasets']['test']['use_aria']
            use_rot= self.opt['datasets']['test']['use_rot']
            input_singleframe = False
            with torch.no_grad():

                if self.L.shape[0] < window_size:


                    if input_singleframe:
                        input_list = []
                        for frame_idx in range(0,self.L.shape[0]):
                            input_list.append(self.L[[frame_idx]].unsqueeze(0))
                        input_tensor = torch.cat(input_list, dim = 0)

                        E_global_orientation_tensor, E_joint_rotation_tensor = self.netG(input_tensor)
                    else:
                        E_global_orientation_list = []                     

                        for frame_idx in range(0,self.L.shape[0]):
                        
                                E_global_orientation = self.netG(self.L[0:frame_idx+1].unsqueeze(0))
                                E_global_orientation_list.append(E_global_orientation)    
                          
                        E_global_orientation_tensor = torch.cat(E_global_orientation_list, dim=0)

                else:  
                    input_list_1 = []
                    input_list_2 = []
                    if input_singleframe:
                        for frame_idx in range(0,window_size):
                            input_list_1.append(self.L[[frame_idx]].unsqueeze(0))
                        input_tensor_1 = torch.cat(input_list_1, dim = 0)
                        E_global_orientation_1, E_joint_rotation_1 = self.netG(input_tensor_1)
                    else:
                        E_global_orientation_list_1 = []          
                        for frame_idx in range(0,window_size):
                            images_input = self.feed_cola(cola, frame_idx, image_transforms, window_size)
                            E_global_orientation = self.netG(self.L[0:frame_idx+1].unsqueeze(0), images_input.unsqueeze(0), do_fk=False)
                            #E_global_orientation = self.netG(images_input.unsqueeze(0), do_fk=False)
                            E_global_orientation_list_1.append(E_global_orientation)
                        E_global_orientation_1 = torch.cat(E_global_orientation_list_1, dim=0)
                    E_global_orientation_list_2 = []
                    for frame_idx in range(window_size,self.L.shape[0]):
                        images_input = self.feed_cola(cola, frame_idx, image_transforms, window_size)
                        E_global_orientation = self.netG(self.L[frame_idx-window_size+1:frame_idx+1,...].unsqueeze(0), images_input.unsqueeze(0), do_fk=False)
                        #E_global_orientation = self.netG(images_input.unsqueeze(0), do_fk=False)
                        E_global_orientation_list_2.append(E_global_orientation)
                    E_global_orientation_2 = torch.cat(E_global_orientation_list_2, dim=0)
                    E_global_orientation_tensor = torch.cat([E_global_orientation_1,E_global_orientation_2], dim=0)
            self.E_global_orientation = E_global_orientation_tensor
            self.E = E_global_orientation_tensor.to(self.device)

            if use_rot:
                self.E = self.E[:,:51]
                
            self.predicted_position = self.E.reshape(-1,17,3)
            self.gt_position = self.gt 
            self.predicted_position = self.predicted_position 

            self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_prediction(self,):
        body_parms = OrderedDict()
        body_parms['position'] = self.predicted_position
        return body_parms

    def current_gt(self, ):
        body_parms = OrderedDict()
        body_parms['position'] = self.gt_position
        return body_parms


    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
