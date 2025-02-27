import torch
import torch.nn as nn
from IPython import embed
import math
from utils import utils_transform
from models.video_transformer import SpaceTimeTransformer

nn.Module.dump_patches = True




class EgoCast(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, device,opt):
        super(EgoCast, self).__init__()

        self.linear_embedding = nn.Linear(input_dim,embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        

        
        self.include_video = opt['netG']['video_model']
        if opt['datasets']['train']['use_rot']:
            out_dim = 58
        else:
            out_dim =51
        if self.include_video:
            self.video_model = SpaceTimeTransformer(num_frames=opt['datasets']['train']['window_size']+1)
            self.stabilizer = nn.Sequential(
                            nn.Linear(768+embed_dim, 256), #embed_dim+768
                            nn.ReLU(),
                            nn.Linear(256, out_dim)
            )
        else:
            self.stabilizer = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, out_dim)
            )
            self.joint_rotation_decoder = nn.Sequential(
                             nn.Linear(embed_dim, 256),
                             nn.ReLU(),
                             nn.Linear(256, 126)
             )

        #self.body_model = body_model

    def forward(self, input_tensor,image=None, do_fk = True):
    #def forward(self, image, do_fk = True):
        input_tensor = input_tensor.reshape(input_tensor.shape[0],input_tensor.shape[1],-1)
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)[:, -1]
        if self.include_video:
            a = self.video_model(image)
            x_mixed = torch.cat([x,a],axis=1)
            global_orientation = self.stabilizer(x_mixed)
            #global_orientation = self.stabilizer(a)
        else:
            global_orientation = self.stabilizer(x)
        return global_orientation