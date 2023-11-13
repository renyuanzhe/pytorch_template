from torch.utils.data import Dataset, DataLoader

import torch
import os
import pickle
import cv2
import numpy as np

from src.config import parse_configs







class H2ODataset(Dataset):
    def __init__(self,phase,root_dir,transform=None, data_cfg=None,model_cfg=None):#phase规定读取哪一部分数据 train,val,test
        super().__init__()






    

    def load_data(self):

        pass

    def __getitem__(self, index):

        pass
    def __len__(self):
        return




def h2o_build_dataloader(cfg,phase='trainval',trainval_ratio=0.7):
    
    data_path=os.path.join(cfg.DATA.data_path, cfg.DATA.dataset)
    transform = None, None
    if cfg.DATA.transform is not None:
        input_size = cfg.DATA.transform.input_size
        transform= Compose([Resize(input_size), ClipToTensor(), Normalize(mean=cfg.DATA.transform.means, std=cfg.DATA.transform.stds)])

    if phase=='trainval':
        trainval_dataset=H2ODataset(phase='trainval',root_dir=data_path,transform=transform,data_cfg=cfg.DATA,model_cfg=cfg.MODEL)

        train_size = int(trainval_dataset.__len__() * trainval_ratio)
        val_size = len(trainval_dataset) - train_size
        print("Number of train/val: {}/{}".format(train_size, val_size))

        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

        train_loader=DataLoader(train_dataset,batch_size=cfg.TRAIN.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.batch_size, shuffle=True,num_workers=cfg.num_workers, pin_memory=True)

        return train_loader, val_loader
    else:

        test_dataset = H2ODataset(phase='test', root_dir=data_path, transform=transform, data_cfg=cfg.DATA,
                                      model_cfg=cfg.MODEL)

        test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True)
        print("Number of test samples: {}".format(test_dataset.__len__()))
        return test_loader


if __name__ == '__main__':

    from tqdm import tqdm

    class data_cfg:pass
    data_cfg.max_frames=20
    data_cfg.load_all=True

    class model_cfg:pass
    model_cfg.target = '2d'
    model_cfg.modalities = ['rgb', 'loc']
    model_cfg.use_global = True
    model_cfg.centralize = True
    model_cfg.normalize = True




    h2o_train_dataset=H2ODataset(phase='train',root_dir='data\H2O\Ego3DTraj-part',transform=rgb_transform,data_cfg=data_cfg,model_cfg=model_cfg)

    train_sample=h2o_train_dataset.__getitem__(0)

    #########################################################
    cfg = parse_configs(phase='train')
    train_loader,val_loader=h2o_build_dataloader(cfg=cfg,phase='trainval',trainval_ratio=0.7)

    for i,(file_path, video, cam_pose, frame_num, traj) in enumerate(train_loader):
        pass










