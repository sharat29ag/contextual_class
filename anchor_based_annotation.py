import argparse
import os
import random
import shutil
import sys
from math import log2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)
import glob
from tqdm import tqdm
from data import create_dataset
from utils.utils import get_logger
from models.adaptation_model import CustomModel
from metrics import runningScore, averageMeter
from loss import get_loss_function
from tensorboardX import SummaryWriter
from scipy.stats import entropy
from collections import Counter

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def average2(f,target_label,num_classes=19):
    a,b,c = f.shape
    f = np.reshape(f,(a,b*c))
    o = np.asarray(np.argmax(f, axis=0), dtype=np.uint8)
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

def annotator(softmax,class_annotate,gt_label,color):
    mapping = {0:7,1:8,2:11,3:12,4:13,5:17,6:19,7:20,8:21,9:22,10:23,11:24,12:25,13:26,14:27,15:28,16:31,17:32,18:33,250:0}

    logits = softmax.detach().cpu().numpy()
    label = np.argmax(logits,axis=1)
    
    label = np.ones_like(label)*250

    for i in range(19):
        index_to_annotate = gt_label == i
        if i in class_annotate:
            label[index_to_annotate] = gt_label[index_to_annotate]
        else:
            label[index_to_annotate] = 250

    if color==0:
        k = np.array(list(mapping.keys()))
        v = np.array(list(mapping.values()))

        out = np.zeros_like(label)
        for key,val in zip(k,v):
            out[label==key] = val
        label = out
    return label

def save_color_mask(label,target_img_name,folder):
    output_col = colorize_mask(label.squeeze())
    folder_name = target_img_name[0].split('/')[-2]
    path = './'+folder
    if not os.path.exists(path):
        os.makedirs(path)
    output_col.save('%s/%s_color.png' % (path, target_img_name[0].split('/')[-1].split('.')[0]))

def save_mask(label,target_img_name,folder,path):
    label = Image.fromarray(label.squeeze().astype(np.uint8)).convert('P')
    folder_name = target_img_name[0].split('/')[-2]
    path = path + str(folder) + '/data/gtFine/train/' + folder_name
    if not os.path.exists(path):
        os.makedirs(path)
    name = target_img_name[0].split('/')[-1].split('.')[0].split('_')
    name = name[0]+'_'+name[1]+'_'+name[2]
    label.save('%s/%s_gtFine_labelIds.png' % (path, name))

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def anchor_based_annotation(outputs_softmax,anchors,target_label):

    output = outputs_softmax.squeeze().cpu().data.numpy()
    output = average2(output,target_label.squeeze(),19)
    dist = np.zeros((19))
    for i in range(19):
        if (np.sum(output[i])) == 0:
            dist[i]=0
        else:
            dist[i] = np.linalg.norm(anchors[i]-output[i])
    dist[dist < np.nanmean(dist)] = 0
    class_annotate = list(np.where(dist!=0))[0]
    return class_annotate

def iou_based_annotation(name,skyline, base):

    name = name.split('_')
    name = name[0]+'_'+name[1]+'_'+name[2]
    skyline = np.load(os.path.join(skyline,name+'.npy'))
    base = np.load(os.path.join(base,name+'.npy'))
    diff = np.zeros((19))

    for i in range(19):
        if skyline[i] == 'nan' and base[i] == 'nan':
            continue
        else:
            if skyline[i] == 'nan':
                diff[i] = base[i]
            elif base[i] == 'nan':
                diff[i] = skyline[i]
            else:
                diff[i] = skyline[i] - base[i]

    class_annotate = [index for (index, number) in enumerate(diff) if number > 0.05]
    return class_annotate


def augmentation_based_annotation(name,raw_f,aug_f):
    
    raw = np.load(os.path.join(raw_f,name+'.npy'))
    aug = np.load(os.path.join(aug_f,name+'.npy'))

    raw = average2(raw,raw.squeeze(),19)
    aug = average2(aug,aug.squeeze(),19)
    
    dist = np.zeros((19))
    for j in range(19):
        dist[j] = np.linalg.norm(raw[j]-aug[j]) ## euclidean distance
        # dist[j] = KL(raw[j],aug[j]) ## KL divergence  
    
    class_annotate = [index for (index, number) in enumerate(dist) if number > 0.05]
    return class_annotate



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
    target_train_loader = datasets.target_train_loader
    logger.info('target train batchsize is {}'.format(target_train_loader.batch_size))
    print('target train batchsize is {}'.format(target_train_loader.batch_size))

    

    file_selected = open('./CDAL_selection.txt') ## path to selection file
    temp = file_selected.readlines()
    names = []
    for i in temp:
        i = i.strip()
        names.append(i.split('/')[-1].split('.')[0])

    # anchors = torch.load('/home/sharat/sharat/new_begin/class_anchors_analysis/target_19x19_58.3_GT_anchors.npy').squeeze()
    anchors = torch.load('./anchors.npy').squeeze() ## path to anchors

    ## path for augmentation based annotation

    # path_raw_f = '/home/sharat/sharat/new_begin/augmentation_based_class_selection/raw_target_new'
    # path_aug_f = '/home/sharat/sharat/new_begin/augmentation_based_class_selection/augmented_features_rotate_5_bright_satur_cont_'
    
    ## path for iou based annotation

    # skyline_path = '/home/sharat/sharat/MADA_1/mIoU_analysis/supervised_numpy'
    # base_path = '/home/sharat/sharat/new_begin/weights/1914x1024/aug_based_annotation/iou_based_feat'


    # os.system('rm -r /home/sharat/sharat/new_begin/weights/1914x1024/aug_based_annotation/sat_cont_rotate/data/gtFine/train')

    
    with torch.no_grad():
        for target_image, target_label, target_img_name in tqdm(datasets.target_train_loader):
            target_image = target_image.to(device)
            if cfg['training'].get('freeze_bn') == True:
                model.freeze_bn_apply()
            if model.PredNet.training:
                model.PredNet.eval()

            name = target_img_name[0].split('/')[-1].split('.')[0]
            if name in names:

                _, _, feat_cls, output = model.PredNet_Forward(target_image)
                outputUp = F.interpolate(output, size=target_image.size()[2:], mode='bilinear', align_corners=True)
                outputs_softmax = F.softmax(outputUp, dim=1)

                # class_annotate  = augmentation_based_annotation(name,path_raw_f,path_aug_f)
                class_annotate = anchor_based_annotation(outputs_softmax,anchors,target_label)
                # class_annotate = iou_based_annotation(name,skyline_path,base_path)

                label = annotator(outputs_softmax,class_annotate,target_label.detach().cpu().numpy(),color=0)
                save_mask(label,target_img_name,'anchor_based_annotation/Cityscapes','./')
                label = annotator(outputs_softmax,class_annotate,target_label.detach().cpu().numpy(),color=1)
                save_color_mask(label,target_img_name,'anchor_based_annotation_color')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",nargs="?",type=str,default='configs/anchor_mask.yml',help="Configuration file to use")

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    logdir = os.path.join('./runs/to_delete')
    writer = SummaryWriter(log_dir=logdir)
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    logger.info('Let the games begin')
    Savefeat(cfg, writer, logger)
