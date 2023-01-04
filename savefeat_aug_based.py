import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import entropy
from collections import Counter

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from tqdm import tqdm
from data import create_dataset
from utils.utils import get_logger
from models.adaptation_model import CustomModel
from metrics import runningScore, averageMeter
from loss import get_loss_function
from tensorboardX import SummaryWriter

def average2(f,target_label,num_classes=19):
    a,b,c=f.shape
    f=np.reshape(f,(a,b*c))
    # o = np.asarray(np.argmax(f, axis=0), dtype=np.uint8)
    o = np.asarray(target_label.flatten())
    w = -1*np.nansum(f*np.log(f),axis=0)
    w = w.T
    fc=[]
    for i in range(num_classes):
        c=np.where(o==i)[0]
        fc1=f[:,np.array(c)]
        wc1=w[np.array(c)]
        wc1/=np.sum(wc1)
        fc1=np.dot(fc1,wc1)
        fc.append(fc1)
    fc=np.stack(fc)
    return fc


def Savefeat(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    ## create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(cfg, writer, logger)  #source_train\ target_train\ source_valid\ target_valid + _loader

    model = CustomModel(cfg, writer, logger)

    # We set the Cityscapes as the target dataset
    target_train_loader = datasets.target_train_loader
    logger.info('target train batchsize is {}'.format(target_train_loader.batch_size))
    print('target train batchsize is {}'.format(target_train_loader.batch_size))

    val_loader = None
    if cfg.get('valset') == 'gta5':
        val_loader = datasets.source_valid_loader
        logger.info('valset is gta5')
        print('valset is gta5')
    else:
        val_loader = datasets.target_valid_loader
        logger.info('valset is cityscapes')
        print('valset is cityscapes')
    logger.info('val batchsize is {}'.format(val_loader.batch_size))
    print('val batchsize is {}'.format(val_loader.batch_size))

    i_iter = 0

    file_selected = open(cfg['training']['active_file_path'])
    temp = file_selected.readlines()
    names = []
    for i in temp:
        i = i.strip()
        names.append(i.split('/')[-1].split('.')[0])

    with torch.no_grad():
        for target_image, target_label, target_img_name in tqdm(datasets.target_train_loader):
            target_image = target_image.to(device)
            if cfg['training'].get('freeze_bn') == True:
                model.freeze_bn_apply()
            if model.PredNet.training:
                model.PredNet.eval()

            _, _, feat_cls, output = model.PredNet_Forward(target_image)

            ##########     Our features   #########
            name = target_img_name[0].split('/')[-1].split('.')[0]
            if name in names:

                outputUp = F.interpolate(output, size=target_image.size()[2:], mode='bilinear', align_corners=True)
                outputs_softmax = F.softmax(outputUp, dim=1)
                output = outputs_softmax.squeeze().cpu().data.numpy()
                output = average2(output,target_label.squeeze(),19)
    
                path = cfg['training']['path_for_features']
                if not os.path.exists(path):
                    os.makedirs(path)
                path = path + '/' + target_img_name[0].split('/')[-1].split('.')[0] + '.npy'
                np.save(path,output)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",nargs="?",type=str,default='configs/savefeat_anchor.yml',help="Configuration file to use")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
   
    logdir = os.path.join('runs')
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    Savefeat(cfg, writer, logger)
