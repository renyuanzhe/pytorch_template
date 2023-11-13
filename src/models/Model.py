import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.models.Losses import CoordLoss


from src.config import parse_configs
from src.datasets.h2o_dataset import h2o_build_dataloader




class Model(nn.Module):
    def __init__(self,cfg,input_img_size=64,sample_frame_num=40) :  #cfg是cfg.MODEL
        super().__init__()

        
 

    def forward(self,video,traj,frame_num,ratios):
        """

        """

        




    