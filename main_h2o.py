import importlib
import torch
import numpy as np

from src.config import parse_configs
from src.models.optimizers import *
from src.datasets.h2o_dataset import h2o_build_dataloader
from src.utils.util import *

from src.trainer.trainer import *

from tensorboardX import SummaryWriter


def test (cfg):
    # build test dataloaders
    print("Loading dataset...")
    test_loader = h2o_build_dataloader(cfg, phase='test')

    # build the model
    model_module = importlib.import_module('src.models.{}'.format(cfg.MODEL.arch))
    model = getattr(model_module, cfg.MODEL.arch)(cfg.MODEL, input_img_size=cfg.DATA.transform.input_size[0],sample_frame_num=cfg.DATA.max_frames)
    model = model.to(device=cfg.device)
    # load checkpoints
    model, test_epoch = load_checkpoint(cfg, model)
    model = model.eval()
    
    # result folder
    result_path = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'test-e{}'.format(test_epoch))
    os.makedirs(result_path, exist_ok=True)
    eval_space = getattr(cfg.TEST, 'eval_space', '3d')

    ### test on the seen scenes
    all_preds, all_gt, all_cam_poses = test_h2o(cfg, model, test_loader, os.path.join(result_path, 'test_results.npz'))
    
    # 计算ade,fde
    all_ades, all_fdes = compute_displacement_errors(all_preds, all_gt, all_cam_poses,
                                                               target=model.target, eval_space=eval_space, use_global=cfg.MODEL.use_global)
    #计算预测轨迹和真实轨迹在 x, y, z 三个维度上的预测值和真实值在各个维度上的平均绝对差。
    all_dxs, all_dys, all_dzs = compute_block_distances(all_preds, all_gt, all_cam_poses,
                                                               target=model.target, eval_space=eval_space, use_global=cfg.MODEL.use_global)
    
    # print tables
    print_de_table(all_ades, all_fdes, subset='test')
    print_delta_table(all_dxs, all_dys, all_dzs, subset='test')

    print("\nDone!")




def train(cfg):
    #本次训练模型文件的保存目录
    model_dir=os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'snapshot')
    ensure_dir(model_dir)

    # tensorboard logging
    logs_dir=os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'logs')#tensorboard日志文件的保存目录
    ensure_dir(logs_dir)
    writer=SummaryWriter(logs_dir)#创建一个SummaryWriter实例


    # build data loaders
    print("Loading dataset...")
    traindata_loader,valdata_loader=h2o_build_dataloader(cfg=cfg,phase='trainval',trainval_ratio=0.7)#trainval_ratio是训练集中用来训练和验证的比例


    # build model
    model_module = importlib.import_module('src.models.{}'.format(cfg.MODEL.arch))#导入模型py文件
    model = getattr(model_module, cfg.MODEL.arch)(cfg.MODEL, input_img_size=cfg.DATA.transform.input_size[0],sample_frame_num=cfg.DATA.max_frames)#实例化模型
    model.train()#设置为训练模式
    model.to(cfg.device)#将模型放到GPU上

    # build the loss criterion
    import src.models.Losses as loss_module#导入损失函数py文件
    criterion = getattr(loss_module, cfg.TRAIN.loss.type)(cfg_train_loss=cfg.TRAIN.loss)#实例化损失函数
    criterion = criterion.to(device=cfg.device)

    # build optimizer  &  lr scheduler
    optimizer=get_optimizer(cfg.TRAIN, model.parameters())#实例化优化器
    scheduler = get_scheduler(cfg.TRAIN, optimizer)

    # training loop
    for epoch in range(cfg.TRAIN.epoch):
        # train one epoch
        avg_batch_loss=train_one_epoch(cfg, model, traindata_loader, criterion, optimizer, writer, epoch)
        
        writer.add_scalar('avg_batch_loss_epoch', avg_batch_loss, epoch)

        if (epoch + 1) % cfg.TRAIN.eval_interval == 0 or epoch  == cfg.TRAIN.epoch-1:# 在验证集上进行评估
            with torch.no_grad(): 
                all_preds_val, all_gt_val, all_cam_poses = eval_h2o(cfg, model, valdata_loader, criterion, writer, epoch)#测试集上的轨迹预测，gt，相机位姿
                all_ades_val, all_fdes_val = compute_displacement_errors(all_preds_val, all_gt_val, all_cam_poses, use_global=cfg.MODEL.use_global)
        
                print_eval_results(writer, all_ades_val, all_fdes_val, epoch=epoch, loss_train=avg_batch_loss)#打印整个验证集上评估结果


        if (epoch+1)%cfg.TRAIN.snapshot_interval==0 or epoch==cfg.TRAIN.epoch-1:#保存模型
            save_dict={'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict()}
            model_file_path=os.path.join(model_dir,cfg.TRAIN.snapshot_prefix + '%02d.pth'%(epoch + 1))
            save_the_latest(save_dict, model_file_path, topK=20, ignores=getattr(cfg.TRAIN.scheduler, 'lr_decay_epoch', []))#只保留最新的 topK 个检查点


        # update learning rate
        scheduler.step(epoch=epoch)

    writer.close()   



if __name__ == '__main__':
    cfg=parse_configs()

    # fix random seed 
    set_deterministic(cfg.seed)

    if cfg.test:
        test(cfg)
        
    else:
        train(cfg)