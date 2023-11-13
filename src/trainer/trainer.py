import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from src.utils.util_h2o import *
import src.utils.util_h2o as util_h2o
from src.utils.specific_util import *
from src.utils.util import *


def train_one_epoch(cfg, model, traindata_loader, criterion, optimizer, writer, epoch):#训练过程对于不同数据集通用
    model.train()#在每个 epoch 的开始都会调用 model.train()，确保模型处于训练模式。然后，在进行验证或测试之前，我们会调用 model.eval() 将模型设置为评估模式。

    pbar = tqdm(total=len(traindata_loader), ncols=0, desc='train epoch {}/{}'.format(epoch + 1, cfg.TRAIN.epoch))
    loss = 0 #loss是此epoch的总损失
    
    for batch_id, batch_data in enumerate(traindata_loader):
        _, clip, _, nframes, traj_gt = batch_data #clip是输入的图像帧序列,odometry是坐标系转换的参数
        # send data to device 
        clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=False)
        
        # generate a batch of random observation ratios
        ratios = random_ratios_batch(nframes, cfg.TRAIN)#(B, N)
        
        # run inference
        outputs = model(clip,traj_gt,nframes,ratios) #output是一个张量，第一个元素是预测的未观察轨迹信息，第二个元素是预测的观察部分的轨迹信息
        
        # compute losses
        output_predict,output_observed = outputs
        losses = criterion(output_predict,output_observed, nframes, traj_gt)#losses在gpu上,本batch的各loss

        loss += losses['total_loss'].item() #.item()解决内存泄漏。loss是此epoch的累计loss

        # backward
        optimizer.zero_grad()#在反向传播之前，需要使用优化器的 zero_grad 方法将梯度清零。这是因为 PyTorch 默认会累积梯度
        losses['total_loss'].backward()
        optimizer.step()#更新模型参数

        # write the lr and losses
        total_batch_num=epoch * len(traindata_loader) + batch_id#自训练开始的batch的总数
        #writer.set_step(total_batch_num)
        writer.add_scalar('lr_batch', optimizer.state_dict()['param_groups'][0]['lr'], total_batch_num)#将当前的学习率添加到 tensorboard 的日志中
        for k, v in losses.items():
            writer.add_scalars('train/{}_batch'.format(k), {k: v}, total_batch_num)  # draw loss curves in different figures
        
        pbar.set_postfix({"this_batch_per_timestep_train loss": losses['total_loss'].item()})#显示本batch的各样本的每个timestep的平均loss
        pbar.update()

    avg_batch_loss = loss/len(traindata_loader)
    pbar.close()
    return avg_batch_loss#返回此epoch的总损失






def eval_h2o(cfg, model, valdata_loader, criterion, writer, epoch):#在整个验证集上实验。验证过程对于不同数据集不通用，因为H2O的验证过程需要用到相机位姿
    model.eval()
    pbar = tqdm(total=len(valdata_loader), ncols=0, desc='eval epoch {}'.format(epoch + 1))

    all_preds, all_gts, all_poses = {}, {}, {}

    for batch_id, batch_data in enumerate(valdata_loader):
        filename, clip, campose, nframes, traj_gt = batch_data
        # send data to device 
        clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=True)
        
        # fixed set of observation ratios
        ratios = get_test_ratios(cfg.TEST.ratios, nframes)#(B, N)
        
        # run inference
        outputs = model(clip,traj_gt,nframes,ratios)

        # compute losses
        output_predict,output_observed = outputs
        losses = criterion(output_predict,output_observed, nframes, traj_gt)#本batch的各loss
   

        # write the lr and losses
        total_batch_num = epoch * len(valdata_loader) + batch_id
        for k, v in losses.items():
            writer.add_scalars('test/{} _batch'.format(k), {k: v}, total_batch_num)#每个batch的loss

        # gather unobserved predictions and ground truths 只计算未观察到的那一部分的预测和gt，是对预处理过程的逆操作，得到真实值
        preds, gts, poses = util_h2o.gather_eval_results(outputs[0], nframes, traj_gt, campose, ignore_depth=model.ignore_depth, 
                                                    use_global=cfg.MODEL.use_global, centralize=model.centralize)#preds是一个字典，键是不同的观察比例，值是一个列表，列表中的元素是各预测的未观察轨迹(T_predict,3)
        
        if batch_id == 0:
            all_preds.update(preds)#all_preds也是一个字典，键是不同的观察比例，值是一个列表，列表中的元素是各预测的未观察轨迹(N,3)
            all_gts.update(gts)
            all_poses.update(poses)
        else:
            for r in list(preds.keys()):
                all_preds[r].extend(preds[r])#在相同的键后扩展value的内容
                all_gts[r].extend(gts[r])
                all_poses[r].extend(poses[r])
        
        pbar.set_postfix({"batch_val loss": losses['total_loss'].item()})#显示本batch的loss
        pbar.update()
    
    for r in list(all_preds.keys()):
        all_preds[r] = np.vstack(all_preds[r])  # 把各比例对应的list中的各轨迹合并成一个长的轨迹   ， all_preds[r] 是(N_long, 3)
        all_gts[r] = np.vstack(all_gts[r])  # (N_long, 3)
        all_poses[r] = np.concatenate(all_poses[r])  # (N_long, 4, 4),每个轨迹每个时间点都有一个不同的变换矩阵
    
    pbar.close()
    return all_preds, all_gts, all_poses


def test_h2o(cfg, model, test_loader, result_file):#在整个验证集上实验。验证过程对于不同数据集不通用，因为H2O的测试过程需要用到相机位姿
    if not os.path.exists(result_file):
        # run test inference
        with torch.no_grad():
            all_preds, all_gts, all_poses = {}, {}, {}
            for batch_id, batch_data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Run testing'):
                
                filename, clip, campose, nframes, traj_gt = batch_data
                # send data to device 
                clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=True)
                
                # fixed set of observation ratios
                ratios = get_test_ratios(cfg.TEST.ratios, nframes)
                
                # run inference
                #outputs = model.inference(clip, nframes, ratios, traj=traj_gt)
                outputs = model(clip,traj_gt,nframes,ratios)
                # gather unobserved predictions and ground truths 
                # Note: the preds and gts are in Global 3D Space for target == '3d, or in pixel space of the 1st frame
                preds, gts, poses = util_h2o.gather_eval_results(outputs[0], nframes, traj_gt, campose, ignore_depth=model.ignore_depth, 
                                                            use_global=cfg.MODEL.use_global, centralize=model.centralize)
                if batch_id == 0:
                    all_preds.update(preds)
                    all_gts.update(gts)
                    all_poses.update(poses)
                else:
                    for r in list(preds.keys()):
                        all_preds[r].extend(preds[r])
                        all_gts[r].extend(gts[r])
                        all_poses[r].extend(poses[r])
            
            for r in list(all_preds.keys()):
                all_preds[r] = np.vstack(all_preds[r])  # (N, 3)
                all_gts[r] = np.vstack(all_gts[r])  # (N, 3)
                all_poses[r] = np.concatenate(all_poses[r])  # (N, 4, 4)



        # save predictions
        np.savez(result_file[:-4], pred=all_preds, gt=all_gts, campose=all_poses)
    else:
        print("Result file exists. Loaded from file: %s."%(result_file))
        all_results = np.load(result_file, allow_pickle=True)
        all_preds, all_gts, all_poses = all_results['pred'][()], all_results['gt'][()], all_results['campose'][()]
    return all_preds, all_gts, all_poses