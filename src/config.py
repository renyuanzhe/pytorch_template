import argparse
import yaml
from easydict import EasyDict
from pprint import pformat
import os
import torch
import datetime

def parse_configs(phase='train'):
    parser = argparse.ArgumentParser()#使用内置的 argparse 模块来处理命令行参数

    parser.add_argument('--config', type=str, default='./config/h2o/usst_res18_3d.yml',
                        help='The relative path of dataset.')
    parser.add_argument('--gpus', type=str, default="0",
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--test', action='store_true',
                        help='If specified, run the evaluation only.')#如果指定了这个参数，就只进行测试，不进行训练
    parser.add_argument('--tag', type=str, default=datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
                        help='The tag to save model results')

    args = parser.parse_args()

    if args.test:
        phase = 'test'
        
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = EasyDict(yaml.safe_load(f))#将配置文件中的配置载入cfg

    cfg.update(vars(args))#将args对象的属性和属性值更新到cfg字典中。

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device =torch.device('cpu')
    cfg.update(device=device)
    cfg.MODEL.update(device=device)

    # add root path
    root_path = os.path.dirname(__file__)
    cfg.update(root_path=root_path)


    # save configs to file
    if phase == 'train':
        exp_dir = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, 'config_{}.yaml'.format(phase)), 'w') as f:
            f.writelines(pformat(vars(cfg)))

    return cfg

if __name__ == '__main__':
    cfg=parse_configs()
    print(111)