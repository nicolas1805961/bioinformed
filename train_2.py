from ast import arg, parse
from email.policy import default
from nbformat import write
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
# import torchgeometry as tgm
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict

from network import *
from dataset import *
from helper import *
from loss import *
from visualize import func_ACDC2D_image_check_dict, func_ACDC2D_training_visual_check
import time
import os
import argparse
from tensorboardX import SummaryWriter
import logging
import PIL
from torchvision.transforms import ToTensor
import warnings
from time import strftime
import sys

#local path
#sys.path.append(r'C:\Users\Portal\Documents\Isensee\nnUNet\nnunet')
#linux path
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'Isensee_unlabeled/nnunet'))

from nnunet.training.network_training.processor2 import Processor2
from nnunet.lib.utils import ConvBlocks2DGroup
from nnunet.lib.training_utils import read_config, build_2d_model
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoaderPreprocessed
from ruamel.yaml import YAML

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def read_config_video(filename):
    yaml = YAML()
    with open(filename) as file:
        config = yaml.load(file)

    return config


#warnings.filterwarnings("ignore")

ce_loss = nn.BCEWithLogitsLoss()

# path for visual check
seg_loss_visual_check_path = './results/visual_check_seg_loss'
general_loss_visual_check_path = './results/visual_check'

if not os.path.exists(seg_loss_visual_check_path):
    os.makedirs(seg_loss_visual_check_path)

def segTransform(mov_mask, dvf):
    grid = generate_grid(mov_mask.type(torch.cuda.FloatTensor), dvf)
    moved_mask = F.grid_sample(mov_mask.type(torch.cuda.FloatTensor), grid, mode='bilinear')

    return moved_mask

def train(optimizer, model, VAE_model, total_loss_list, reg_loss_list, seg_loss_list, training_data_loader, losstype, dataset, batch_size, n_epoch, learning_rate, lmbd, nup, gamma, save_train_fig, log_path, current_epoch):

    # check input argument
    assert dataset in ['ACDC17', 'Lib']
    assert losstype in ['vae', 'l2', 'noreg', 'bmreg', 'bmreg_seg', 'bmreg_ceseg']

    # parameter configuration
    lr = learning_rate
    bs = batch_size
    n_epoch = n_epoch
    # base_err = 100

    flow_criterion = nn.MSELoss()
    Tensor = torch.cuda.FloatTensor

    epoch_loss = []

    if losstype == 'vae':
        VAE_epoch_loss = []
    if losstype == 'l2':
        l2_norm_epoch_loss = []
    if losstype == 'bmreg' or losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
        bm_reg_epoch_loss = []
    if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
        bmreg_seg_epoch_loss = []

    for batch_idx in tqdm(range(1, 251)): # tqdm progress bar
        data_dict = next(training_data_loader)
        mov = data_dict['unlabeled'][-1]
        fix = data_dict['unlabeled'][0]
        mov_seg = data_dict['target'][-1].short()
        fix_seg = data_dict['target'][0].short()

        for b in range(len(mov)):
            my_min = min(mov[b].min(), fix[b].min())
            my_max = max(mov[b].max(), fix[b].max())
            mov[b] = (mov[b] - my_min) / (my_max - my_min + 1e-8)
            fix[b] = (fix[b] - my_min) / (my_max - my_min + 1e-8)

        #print(mov.shape)
        #print(fix.shape)
        #print(mov_seg.shape)
        #print(fix_seg.shape)
#
        #fig, ax = plt.subplots(1, 4)
        #ax[0].imshow(mov_seg[0, 0].cpu(), cmap='gray')
        #ax[1].imshow(fix_seg[0, 0].cpu(), cmap='gray')
        #ax[2].imshow(mov[0, 0].cpu(), cmap='gray')
        #ax[3].imshow(fix[0, 0].cpu(), cmap='gray')
        #plt.show()

        # print(x-x_pred)
        mov_c = Variable(mov.type(Tensor))
        fix_c = Variable(fix.type(Tensor))

        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(mov_c[0, 0].cpu(), cmap='gray')
        #ax[1].imshow(fix_c[0, 0].cpu(), cmap='gray')
        #plt.show()

        net = model(mov_c, fix_c, mov_c)
        
        # replace VAE loss with L2-norm
        df_gradient = compute_gradient(net['out'])

        # for tensorboard
        pred_dvf = net['out']
        mov_LV_mask = (mov_seg==3).type(torch.cuda.ByteTensor)
        fix_LV_mask = (fix_seg==3).type(torch.cuda.ByteTensor)

        mov_myo_mask = (mov_seg==2).type(torch.cuda.ByteTensor)
        fix_myo_mask = (fix_seg==2).type(torch.cuda.ByteTensor)

        mov_RV_mask = (mov_seg==1).type(torch.cuda.ByteTensor)
        fix_RV_mask = (fix_seg==1).type(torch.cuda.ByteTensor)

        mov_epi_mask = mov_LV_mask + mov_myo_mask
        fix_epi_mask = fix_LV_mask + fix_myo_mask

        moved_LV_mask = segTransform(mov_LV_mask, pred_dvf)
        moved_RV_mask = segTransform(mov_RV_mask, pred_dvf)
        moved_epi_mask = segTransform(mov_epi_mask, pred_dvf)
        manual_moved = segTransform(mov, pred_dvf)

        if losstype == 'l2':
            l2_reg_loss = torch.norm(df_gradient)
            loss = flow_criterion(net['fr_st'], fix_c) + lmbd * l2_reg_loss
        
        if losstype == 'noreg':
            loss = flow_criterion(net['fr_st'], fix_c)

        if losstype == 'bmreg':
            bmreg_loss = BMIloss(net['out'], nup=nup)
            loss = flow_criterion(net['fr_st'], fix_c) + lmbd * bmreg_loss

        if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
            bmreg_loss = BMIloss(net['out'], nup=nup)
            # pred_dvf = net['out']
            dice_criterion = DiceLoss()

            dice_LV = dice_criterion(moved_LV_mask, fix_LV_mask)
            dice_RV = dice_criterion(moved_RV_mask, fix_RV_mask)
            dice_epi = dice_criterion(moved_epi_mask, fix_epi_mask)

            seg_dice_loss = (dice_LV + dice_RV + dice_epi) / 3

            if losstype == 'bmreg_ceseg':
                ce_loss_LV = ce_loss(moved_LV_mask.float(), fix_LV_mask.float())
                ce_loss_RV = ce_loss(moved_RV_mask.float(), fix_RV_mask.float())
                ce_loss_epi = ce_loss(moved_epi_mask.float(), fix_epi_mask.float())

                ce_seg_loss = (ce_loss_LV + ce_loss_RV + ce_loss_epi)/3

                seg_dice_loss = 0.5 * seg_dice_loss + 0.5 * ce_seg_loss
            # print(seg_dice_loss)

            loss = flow_criterion(net['fr_st'], fix_c) + lmbd * bmreg_loss + gamma * seg_dice_loss
            #loss = seg_dice_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

        if losstype == 'l2':
            l2_norm_epoch_loss.append(l2_reg_loss.item())
        if losstype == 'bmreg' or losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
            bm_reg_epoch_loss.append(bmreg_loss.item())
        if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
            bmreg_seg_epoch_loss.append(seg_dice_loss.item())
        
        ##################
        ## visual check ##
        ##################
        # print(mov.shape)
        # print(fix.shape)
        # print(mov_seg.shape)
        # print(fix_seg.shape)

        if save_train_fig == True:
            if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
                func_ACDC2D_training_visual_check(seg_loss_visual_check_path, batch_idx, mov, fix, manual_moved, mov_LV_mask, mov_RV_mask, mov_epi_mask, fix_LV_mask, fix_RV_mask, fix_epi_mask, moved_LV_mask, moved_RV_mask, moved_epi_mask)
            else:
                loss_visual_check_path = general_loss_visual_check_path + '_{}_loss'.format(losstype)
                if not os.path.exists(loss_visual_check_path):
                    os.makedirs(loss_visual_check_path)
                func_ACDC2D_training_visual_check(loss_visual_check_path, batch_idx, mov, fix, manual_moved, mov_LV_mask, mov_RV_mask, mov_epi_mask, fix_LV_mask, fix_RV_mask, fix_epi_mask, moved_LV_mask, moved_RV_mask, moved_epi_mask)

        ##################
        ## end ##
        ##################
    

    #fig, ax = plt.subplots(1, 1)
    #print(mov.shape)
    #ax.imshow(mov[0, 0].cpu(), cmap='gray')
    #plt.show()
    
    # tensorboard image check (show the last image in the batch)
    image_check_dict = func_ACDC2D_image_check_dict(mov, fix, manual_moved, mov_LV_mask, mov_RV_mask, mov_epi_mask, fix_LV_mask, fix_RV_mask, fix_epi_mask, moved_LV_mask, moved_RV_mask, moved_epi_mask)
    image_check_dict['dvf'] = pred_dvf.detach().cpu().numpy()[0,:,:,:]

        # if batch_idx % 10 == 0:
    # after all batches
    total_loss_list.append(np.mean(epoch_loss))

    if losstype =='vae':
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, VAE_Loss: {:.6f}'.format(n_epoch, batch_idx * len(mov), len(training_data_loader.dataset),100. * batch_idx / len(training_data_loader), np.mean(epoch_loss), np.mean(VAE_epoch_loss)))
        reg_loss_list.append(np.mean(VAE_epoch_loss))

    if losstype == 'l2':
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, L2_loss: {:.6f}'.format(n_epoch, batch_idx * len(mov), len(training_data_loader.dataset),100. * batch_idx / len(training_data_loader), np.mean(epoch_loss), np.mean(l2_norm_epoch_loss)))
        reg_loss_list.append(np.mean(l2_norm_epoch_loss))

    if losstype == 'noreg':  
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(n_epoch, batch_idx * len(mov), len(training_data_loader.dataset),100. * batch_idx / len(training_data_loader), np.mean(epoch_loss)))

    if losstype == 'bmreg':
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, bmi_loss: {:.6f}'.format(n_epoch, batch_idx * len(mov), len(training_data_loader.dataset),100. * batch_idx / len(training_data_loader), np.mean(epoch_loss), np.mean(bm_reg_epoch_loss)))
        reg_loss_list.append(np.mean(bm_reg_epoch_loss))

    if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, bmi_loss: {:.6f}, seg_loss {:.6f}'.format(n_epoch, current_epoch + 1, n_epoch, 100. * (current_epoch + 1) / n_epoch, np.mean(epoch_loss), np.mean(bm_reg_epoch_loss), np.mean(bmreg_seg_epoch_loss)))
        reg_loss_list.append(np.mean(bm_reg_epoch_loss))
        seg_loss_list.append(np.mean(bmreg_seg_epoch_loss))

    return optimizer, model, VAE_model, total_loss_list, reg_loss_list, seg_loss_list, image_check_dict

def validation(model_name, total_loss_list, reg_loss_list, seg_loss_list, validation_data_loader, model, VAE_model, losstype, dataset, bs, n_epoch, lr, lmbd, nup, gamma, save_train_fig, model_save_root, log_path):

    # check input arguments
    assert dataset in ['ACDC17', 'Lib']
    assert losstype in ['vae', 'l2', 'noreg', 'bmreg', 'bmreg_seg', 'bmreg_ceseg']

    # model save path
    # model_save_root = './models'
    if not os.path.exists(model_save_root):
        os.makedirs(model_save_root)

    model_save_path = os.path.join(model_save_root, model_name)


    Tensor = torch.cuda.FloatTensor
    flow_criterion = nn.MSELoss()

    model.eval()
    test_loss = []

    if losstype =='vae':
        VAE_test_loss = []
    
    if losstype == 'l2':
        l2_norm_epoch_loss = []
    
    if losstype == 'bmreg' or losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
        bm_reg_epoch_loss = []
    
    if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
        bmreg_seg_epoch_loss = []

    # global base_err
    base_err = 100

    for batch_idx in tqdm(range(1, 51)): # tqdm progress bar
        data_dict = next(validation_data_loader)
        mov = data_dict['unlabeled'][-1]
        fix = data_dict['unlabeled'][0]
        mov_seg = data_dict['target'][-1].short()
        fix_seg = data_dict['target'][0].short()

        for b in range(len(mov)):
            my_min = min(mov[b].min(), fix[b].min())
            my_max = max(mov[b].max(), fix[b].max())
            mov[b] = (mov[b] - my_min) / (my_max - my_min + 1e-8)
            fix[b] = (fix[b] - my_min) / (my_max - my_min + 1e-8)

        # print(x-x_pred)
        mov_c = Variable(mov.type(Tensor))
        fix_c = Variable(fix.type(Tensor))

        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(mov_c[0, 0].cpu(), cmap='gray')
        #ax[1].imshow(fix_c[0, 0].cpu(), cmap='gray')
        #plt.show()


        net = model(mov_c, fix_c, mov_c)

        df_gradient = compute_gradient(net['out'])


        pred_dvf = net['out']
        mov_LV_mask = (mov_seg==3).type(torch.cuda.ByteTensor)
        fix_LV_mask = (fix_seg==3).type(torch.cuda.ByteTensor)

        mov_myo_mask = (mov_seg==2).type(torch.cuda.ByteTensor)
        fix_myo_mask = (fix_seg==2).type(torch.cuda.ByteTensor)

        mov_RV_mask = (mov_seg==1).type(torch.cuda.ByteTensor)
        fix_RV_mask = (fix_seg==1).type(torch.cuda.ByteTensor)

        mov_epi_mask = mov_LV_mask + mov_myo_mask
        fix_epi_mask = fix_LV_mask + fix_myo_mask

        moved_LV_mask = segTransform(mov_LV_mask, pred_dvf)
        moved_RV_mask = segTransform(mov_RV_mask, pred_dvf)
        moved_epi_mask = segTransform(mov_epi_mask, pred_dvf)
        manual_moved = segTransform(mov, pred_dvf)

        if losstype == 'l2':
            l2_reg_loss = torch.norm(df_gradient)
            loss = flow_criterion(net['fr_st'], fix_c) + lmbd * l2_reg_loss 
        
        if losstype == 'noreg':
            loss = flow_criterion(net['fr_st'], fix_c)

        if losstype == 'bmreg':
            bmreg_loss = BMIloss(net['out'], nup=nup)
            loss = flow_criterion(net['fr_st'], fix_c) + lmbd * bmreg_loss 

        if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
            bmreg_loss = BMIloss(net['out'], nup=nup)
            # pred_dvf = net['out']
            dice_criterion = DiceLoss()

            dice_LV = dice_criterion(moved_LV_mask, fix_LV_mask)
            dice_RV = dice_criterion(moved_RV_mask, fix_RV_mask)
            dice_epi = dice_criterion(moved_epi_mask, fix_epi_mask)

            seg_dice_loss = (dice_LV + dice_RV + dice_epi) / 3
            
            if losstype == 'bmreg_ceseg':
                ce_loss_LV = ce_loss(moved_LV_mask.float(), fix_LV_mask.float())
                ce_loss_RV = ce_loss(moved_RV_mask.float(), fix_RV_mask.float())
                ce_loss_epi = ce_loss(moved_epi_mask.float(), fix_epi_mask.float())

                ce_seg_loss = (ce_loss_LV + ce_loss_RV + ce_loss_epi)/3

                seg_dice_loss = 0.5 * seg_dice_loss + 0.5 * ce_seg_loss

            # print(seg_dice_loss)
            
            # loss = flow_criterion(net['fr_st'], fix_c) + lmbd * bmreg_loss + gamma * seg_dice_loss

            loss = flow_criterion(net['fr_st'], fix_c) + lmbd * bmreg_loss + gamma * seg_dice_loss
            #loss = seg_dice_loss

    test_loss.append(loss.item())

    total_loss_list.append(np.mean(test_loss))

    if losstype =='vae':
        VAE_test_loss.append(VAE_loss.item())
        print('Loss: {:.6f}, VAE_Loss: {:.6f}'.format(np.mean(test_loss), np.mean(VAE_test_loss)))
        reg_loss_list.append(np.mean(VAE_test_loss))

    if losstype == 'l2':
        l2_norm_epoch_loss.append(l2_reg_loss.item())
        print('Loss: {:.6f}, L2_Loss: {:.6f}'.format(np.mean(test_loss), np.mean(l2_norm_epoch_loss)))
        reg_loss_list.append(np.mean(l2_norm_epoch_loss))

    if losstype == 'noreg':
        print('Loss: {:.6f}'.format(np.mean(test_loss)))

    if losstype == 'bmreg':
        bm_reg_epoch_loss.append(bmreg_loss.item())
        print('Loss: {:.6f}, bmi_loss: {:.6f}'.format(np.mean(test_loss), np.mean(bm_reg_epoch_loss)))
        reg_loss_list.append(np.mean(bm_reg_epoch_loss))
        
    if losstype == 'bmreg_seg' or losstype == 'bmreg_ceseg':
        bm_reg_epoch_loss.append(bmreg_loss.item())
        bmreg_seg_epoch_loss.append(seg_dice_loss.item())
        print('Loss: {:.6f}, bmi_loss: {:.6f}, seg_loss: {:.6f}'.format(np.mean(test_loss), np.mean(bm_reg_epoch_loss), np.mean(bmreg_seg_epoch_loss)))
        reg_loss_list.append(np.mean(bm_reg_epoch_loss))
        seg_loss_list.append(np.mean(bmreg_seg_epoch_loss))
    #     print(loss)
    #     print(l2_reg_loss)

    # print('Loss: {:.6f}'.format(np.mean(test_loss)))
        
    torch.save(model.state_dict(), model_save_path)
    print("Checkpoint saved to {}".format(model_save_path))

    #if np.mean(test_loss) < base_err:
    #    torch.save(model.state_dict(), model_save_path)
    #    print("Checkpoint saved to {}".format(model_save_path))
    #    base_err = np.mean(test_loss)
    
    return total_loss_list, reg_loss_list, seg_loss_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--losstype', required=True, type=str, help='(vae or l2 or noreg or bmreg or bmreg_seg or bmreg_ceseg) loss type used in the model')
    #parser.add_argument('--dataset', required=True, type=str, help='(ACDC17 or Lib) which dataset to run')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--epoch', default=180, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lmbd', default=0.05, type=float, help='regularization parameter for reg loss')
    parser.add_argument('--nup', default=0.4, type=float, help='poisson ratio for 2D material property')
    parser.add_argument('--gamma', default=0.01, type=float, help='regularization parameter for seg loss')
    parser.add_argument('--save_train_fig', default=False, type=bool, help='whether to save training figures for visual check')
    parser.add_argument('--model_save_root', default='./models', help='model save path')
    parser.add_argument('--stats_save_path', default='./results_statistics', help='stats (losses) save path')
    parser.add_argument('--log_path', default='./logs', help='log path')
    #parser.add_argument('--all_data', action='store_true', default=None, help='Whether to train with all data or not')
    args = parser.parse_args()


    config = read_config_video(args.config)
    args.epoch = config['epoch']
    gamma = config['seg_weight']
    lmbd = config['regu_weight']

    print('Training configuration', args)
    #if args.dataset == 'ACDC17':
    #    data_path = '../../Dataset/ACDC2017/'
    #    train_set = TrainDatasetACDC(os.path.join(data_path, 'training'))
    #    val_set = TrainDatasetACDC(os.path.join(data_path, 'validation'))
#
    #if args.dataset == 'Lib':
    #    data_path = '../../Dataset/Lib/'
    #    train_set = TrainDatasetACDC(os.path.join(data_path, 'training'))
    #    val_set = TrainDatasetACDC(os.path.join(data_path, 'validation'))

    #linux path
    folder_with_preprocessed_data = os.path.join(os.path.dirname(os.getcwd()), 'Isensee_unlabeled/nnunet/out/nnUNet_preprocessed/Task032_Lib/custom_experiment_planner_stage0')
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'Isensee_unlabeled/nnunet/Lib_resampling_training_mask')
    pkl_path= os.path.join(os.path.dirname(os.getcwd()), 'Isensee_unlabeled/nnunet/custom_lib_t_4')
    splits_file = os.path.join('splits/Lib/val/splits_final.pkl')

    #local path
    #folder_with_preprocessed_data = r'C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task032_Lib\custom_experiment_planner_stage0'
    #data_path = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_training_mask"
    #pkl_path=r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\custom_lib_t_4"
    #splits_file = os.path.join(r'splits\Lib\val\splits_final.pkl')

    dataset = load_dataset(folder_with_preprocessed_data)

    print("Using splits from existing split file:", splits_file)
    splits = load_pickle(splits_file)
    print("The split file contains %d splits." % len(splits))

    tr_keys = splits[0]['train']
    val_keys = splits[0]['val']
    print("This split has %d training and %d validation cases." % (len(tr_keys), len(val_keys)))

    tr_keys.sort()
    val_keys.sort()
    dataset_tr = OrderedDict()
    for i in tr_keys:
        dataset_tr[i] = dataset[i]
    dataset_val = OrderedDict()
    for i in val_keys:
        dataset_val[i] = dataset[i]

    crop_size = 192
    image_size = 384
    window_size = 8
    cropper_weights_folder_path = 'Quorum_cardioTrack_all_phases'
    
    cropper_config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'), False, False)

    #cropping_conv_layer = ConvBlocks2DGroup
    #cropping_network = build_2d_model(cropper_config, conv_layer=cropping_conv_layer, norm=getattr(torch.nn, cropper_config['norm']), log_function=None, image_size=image_size, window_size=window_size, middle=False, num_classes=4, processor=None)
    #cropping_network.load_state_dict(torch.load(os.path.join(cropper_weights_folder_path, 'model_final_checkpoint.model'))['state_dict'], strict=True)
    #cropping_network.eval()
    #cropping_network.do_ds = False
#
    #processor = Processor2(crop_size=crop_size, image_size=image_size, cropping_network=cropping_network)

    patch_size = [384, 384]
    video_length = 2

    validation_data_loader = DataLoaderPreprocessed(dataset_val, patch_size, patch_size, 1, do_data_aug=False, video_length=2, point_loss=False, binary_distance_input=False, binary_distance_loss=False, pkl_path=pkl_path,
                                    crop_size=crop_size, processor=None, is_val=True, distance_map_power=1, data_path=data_path, start_es=False, oversample_foreground_percent=0.33,
                                    pad_mode="constant", pad_sides=None, memmap_mode='r')

    training_data_loader = DataLoaderPreprocessed(dataset_tr, [451, 451], patch_size, args.bs, do_data_aug=True, video_length=2, point_loss=False, binary_distance_input=False, binary_distance_loss=False, pkl_path=pkl_path,
                                            crop_size=crop_size, processor=None, is_val=False, distance_map_power=1, data_path=data_path, start_es=False, oversample_foreground_percent=0.33,
                                            pad_mode="constant", pad_sides=None, memmap_mode='r')

    # loading the data
    #training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.bs, shuffle=True)
    #validation_data_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=args.bs, shuffle=False)

    # training/validation data loader check
    # for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
    #                              total=len(training_data_loader)): # tqdm progress bar
    #     mov, fix, mov_seg, fix_seg, mask = batch # source image, target image, mov_seg, fix_seg, myocardial mask
    #     print(mov.shape)
    #     print(fix.shape)
    #     print(mov_seg.shape)
    #     print(fix_seg.shape)
    #     print(mask.shape)
    #     break

    # train loss list
    train_total_loss_list = []
    train_reg_loss_list = [] # for vae, l2, bmreg, bmreg_seg
    train_seg_loss_list = [] # for bmreg_seg

    valid_total_loss_list = []
    valid_reg_loss_list = [] # for vae, l2, bmreg, bmreg_seg
    valid_seg_loss_list = [] # for bmreg_seg
    
    # tensorboardX configuration
    if args.losstype == 'bmreg':
        model_name = 'model_{}_{}_nup_{}_bs_{}_epoch_{}_lr_{}_lmbd_{}.pth'.format('Lib', str(args.losstype), str(args.nup), str(args.bs), str(args.epoch), str(args.lr), str(lmbd))
    elif args.losstype == 'bmreg_seg' or args.losstype == 'bmreg_ceseg':
        model_name = 'model_{}_{}_nup_{}_bs_{}_epoch_{}_lr_{}_lmbd_{}_gamma_{}.pth'.format('Lib', str(args.losstype), str(args.nup), str(args.bs), str(args.epoch), str(args.lr), str(lmbd), str(gamma))
    else:
        model_name = 'model_{}_{}_bs_{}_epoch_{}_lr_{}_lmbd_{}.pth'.format('Lib', str(args.losstype), str(args.bs), str(args.epoch), str(args.lr), str(lmbd))
    

    log_path_model = os.path.join(args.log_path, model_name.split('.pth')[0])
    if not os.path.exists(log_path_model):
        os.makedirs(log_path_model)
    writer = SummaryWriter(log_path_model)

    # configure

    # check relevant paths
    model_load_path = './bioinformed-vae/models/registration_model_pretrained_0.001_32.pth'
    VAE_model_load_path = './bioinformed-vae/models/VAE_recon_model_pretrained.pth'

    timestr = strftime("%Y-%m-%d_%HH%M")
    save_model_path = os.path.join(args.model_save_root, timestr)
    # load registration model 
    model = Registration_Net()
    model.load_state_dict(torch.load(model_load_path))
    model = model.cuda()

    # if dataset == 'ACDC17' and losstype =='vae':
    VAE_model = MotionVAE2D(img_size=96, z_dim=32)
    VAE_model = VAE_model.cuda()
    VAE_model.load_state_dict(torch.load(VAE_model_load_path))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model.train()

    for epoch in range(args.epoch):
        start = time.time()
        # train(epoch)
        optimizer, model, VAE_model, train_total_loss_list, train_reg_loss_list, train_seg_loss_list, image_check_dict= train(optimizer, model, VAE_model, train_total_loss_list, train_reg_loss_list, train_seg_loss_list, training_data_loader, args.losstype, 'Lib', args.bs, args.epoch, args.lr, lmbd, args.nup, gamma, args.save_train_fig, args.log_path, epoch)
        end = time.time()
        print("training took {:.8f}".format(end-start))

        writer.add_scalar('loss/train_loss', train_total_loss_list[epoch], epoch)
        if not args.losstype == 'noreg':
            writer.add_scalar('loss/train_reg_loss', train_reg_loss_list[epoch], epoch)
        if args.losstype == 'bmreg_seg' or args.losstype == 'bmreg_ceseg':
            writer.add_scalar('loss/train_seg_loss', train_seg_loss_list[epoch], epoch)

        start = time.time()
        valid_total_loss_list, valid_reg_loss_list, valid_seg_loss_list = validation(model_name, valid_total_loss_list, valid_reg_loss_list, valid_seg_loss_list, validation_data_loader, model, VAE_model, args.losstype, 'Lib',  args.bs, args.epoch, args.lr, lmbd, args.nup, gamma, args.save_train_fig, save_model_path, args.log_path)
        end = time.time()
        print("testing took {:.8f}".format(end-start))

        # add losses to tensorboard
        writer.add_scalar('loss/valid_loss', valid_total_loss_list[epoch], epoch)

        if not args.losstype == 'noreg':
            writer.add_scalar('loss/valid_reg_loss', valid_reg_loss_list[epoch], epoch)

        if args.losstype == 'bmreg_seg' or args.losstype == 'bmreg_ceseg':
            writer.add_scalar('loss/valid_seg_loss', valid_seg_loss_list[epoch], epoch)

        # add images to tensorboard
        writer.add_image('images/mov_slice', image_check_dict['mov_slice'], epoch)
        writer.add_image('images/fix_slice', image_check_dict['fix_slice'], epoch)
        writer.add_image('images/moved_slice', image_check_dict['moved_slice'], epoch)

        writer.add_image('images/diff_LV', image_check_dict['diff_LV'], epoch)
        writer.add_image('images/diff_RV', image_check_dict['diff_RV'], epoch)
        writer.add_image('images/diff_epi', image_check_dict['diff_epi'], epoch)

    writer.close()

    # save loss statistics
    loss_save_path = os.path.join(args.stats_save_path, model_name.split('.pth')[0])
    if not os.path.exists(loss_save_path):
        os.makedirs(loss_save_path)
    np.save(os.path.join(loss_save_path, 'train_total_loss.npy'), np.asarray(torch.Tensor(train_total_loss_list).cpu()))
    np.save(os.path.join(loss_save_path, 'train_reg_loss.npy'), np.asarray(torch.Tensor(train_reg_loss_list).cpu()))
    np.save(os.path.join(loss_save_path, 'train_seg_loss.npy'), np.asarray(torch.Tensor(train_seg_loss_list).cpu()))

    np.save(os.path.join(loss_save_path, 'valid_total_loss.npy'), np.asarray(torch.Tensor(valid_total_loss_list).cpu()))
    np.save(os.path.join(loss_save_path, 'valid_reg_loss.npy'), np.asarray(torch.Tensor(valid_reg_loss_list).cpu()))
    np.save(os.path.join(loss_save_path, 'valid_seg_loss.npy'), np.asarray(torch.Tensor(valid_seg_loss_list).cpu()))

