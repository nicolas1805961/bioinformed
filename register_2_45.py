#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import matplotlib.pyplot as plt
import os
import argparse
from monai.transforms import NormalizeIntensity
from tqdm import tqdm
import pickle
from glob import glob
import shutil
from pathlib import Path
import sys

# third party
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

sys.path.append(r'C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\network_architecture')


from nnunet.lib.training_utils import build_flow_model_successive, read_config_video
from nnunet.network_architecture.Optical_flow_model_successive import ModelWrap
from nnunet.network_architecture.integration import SpatialTransformer

from network import *


def segTransform(mov_mask, dvf):
    grid = generate_grid(mov_mask.type(torch.cuda.FloatTensor), dvf)
    moved_mask = F.grid_sample(mov_mask.type(torch.cuda.FloatTensor), grid, mode='bilinear')

    return moved_mask


def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

def inference_iterative(path_list_gz, newpath_flow, newpath_registered, model, image_size):

    patient_name = os.path.basename(path_list_gz[0]).split('_')[0]

    newpath_flow = os.path.join(newpath_flow, patient_name)
    newpath_registered = os.path.join(newpath_registered, patient_name)

    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    x_list = []
    gt_list = []
    mask_list = []
    for path_gz in path_list_gz:
        data = np.load(path_gz)
        x = data[0][None, None]
        gt = data[1][None, None]
        mask = data[2:-1][None]

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(mask[0, 0, :, :, 0], cmap='hot')
        #ax[1].imshow(mask[0, 1, :, :, 0], cmap='hot')
        #ax[2].imshow(mask[0, 2, :, :, 0], cmap='hot')
        #plt.show()

        x = torch.from_numpy(x).to(device).float()
        gt = torch.from_numpy(gt).to(device).float()
        mask = torch.from_numpy(mask).to(device).float()

        #x, fixed_affine = vxm.py.utils.load_volfile(path_gz, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        #gt = vxm.py.utils.load_volfile(path_gt, add_batch_axis=True, add_feat_axis=add_feat_axis)
        
        x_list.append(x)
        gt_list.append(gt)
        mask_list.append(mask)
    
    x = torch.stack(x_list, dim=0)
    gt = torch.stack(gt_list, dim=0)
    mask = torch.stack(mask_list, dim=0)

    #data = []
    #for path in path_list_gz:
    #    current_data = nib.load(path)
    #    nifti_header = current_data.header
    #    nifti_affine = current_data.affine
    #    arr = current_data.get_fdata()
    #    #arr, affine = vxm.py.utils.load_volfile(path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    #    x = torch.from_numpy(arr).to(device).float()
    #    data.append(x)
    #data = torch.stack(data, dim=0)
    ##data = data.permute(5, 0, 1, 2, 3, 4).contiguous()
#
    #data_seg = []
    #for path in path_list_gt:
    #    current_data = nib.load(path)
    #    arr = current_data.get_fdata()
    #    #arr = vxm.py.utils.load_volfile(path, add_batch_axis=True, add_feat_axis=add_feat_axis)
    #    x = torch.from_numpy(arr).to(device).float()
    #    data_seg.append(x)
    #data_seg = torch.stack(data_seg, dim=0)
    ##data_seg = data_seg.permute(5, 0, 1, 2, 3, 4).contiguous()
#
    #data_seg = data_seg.permute(3, 0, 1, 2).contiguous()
    #data = data.permute(3, 0, 1, 2).contiguous() # D, T, H, W

    fixed_path = path_list_gz[0]

    flow_list_all = []
    moved_list_all = []
    img_list_all = []
    for d in range(x.shape[-1]):
        fix_seg = gt[0, :, :, :, :, d]
        current_fixed = x[0, :, :, :, :, d]

        #current_fixed = data[d, 0][None, None]
        #fix_seg = data_seg[d, 0][None, None]
        out_list = []
        out_list_flow = []
        out_list_img = []
        for t in range(len(path_list_gz)):
            moving_path = path_list_gz[t]
            if fixed_path == moving_path:
                continue
            filename = os.path.basename(moving_path)

            current_moving = x[t, :, :, :, :, d]
            mov_seg = gt[t, :, :, :, :, d].short()
            #current_moving = data[d, t][None, None]
            #mov_seg = data_seg[d, t][None, None].short()

            for b in range(len(current_moving)):
                my_min = min(current_moving[b].min(), current_fixed[b].min())
                my_max = max(current_moving[b].max(), current_fixed[b].max())
                current_moving[b] = (current_moving[b] - my_min) / (my_max - my_min + 1e-8)
                current_fixed[b] = (current_fixed[b] - my_min) / (my_max - my_min + 1e-8)
        
            x_in = torch.stack([current_fixed, current_moving], dim=0)

            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(x[0].cpu(), cmap='gray')
            #ax[1].imshow(x[1].cpu(), cmap='gray')
            #plt.show()

            net = model(x_in[1], x_in[0], x_in[1])
            pred_dvf = net['out'].detach()

            class_list = []
            for i in range(4):
                mov_mask = mov_seg == i

                grid = generate_grid(mov_mask.float(), pred_dvf)
                moved_mask = F.grid_sample(mov_mask.float(), grid, mode='bilinear')

                #fig, ax = plt.subplots(1, 1)
                #ax.imshow(moved_mask[0, 0, :, :].cpu(), cmap='gray')
                #plt.show()

                class_list.append(moved_mask)
            
            #moved_mask = torch.argmax(torch.cat(class_list, dim=1), dim=1)[0]
            moved_mask = torch.cat(class_list, dim=1)[0]

            out_list.append(moved_mask)
            out_list_flow.append(torch.flip(pred_dvf[0].permute(1, 2, 0).contiguous(), dims=[-1]))
            out_list_img.append(current_moving[0, 0])
        
        moved = torch.stack(out_list, dim=0).detach().cpu().numpy() # T, C, H, W
        flow = torch.stack(out_list_flow, dim=0).detach().cpu().numpy()
        img_sequence = torch.stack(out_list_img, dim=0).detach().cpu().numpy()

        flow_list_all.append(flow)
        moved_list_all.append(moved)
        img_list_all.append(img_sequence)
    
    moved = np.stack(moved_list_all, axis=-1) # T-1, C, H, W, D
    img = np.stack(img_list_all, axis=-1) # T-1, H, W, D
    flow = np.stack(flow_list_all, axis=-2) # T-1, H, W, D, 2

    flow = flow * (image_size / 2)

    #fig, ax = plt.subplots(1, 3)
    #ax[0].imshow(img[10, :, :, 0], cmap='gray')
    #ax[1].imshow(moved[10, :, :, 0], cmap='gray')
    #ax[2].imshow(data[0, 0].cpu(), cmap='gray')
    #plt.show()

    for t in range(len(moved)):
        # save moved image
        moving_path = path_list_gz[t + 1]

        filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path = os.path.join(newpath_registered, filename)
        np.savez(save_path, seg=moved[t].squeeze())
        #saved_image = nib.Nifti1Image(moved[t].squeeze(), affine=nifti_affine, header=nifti_header)
        #nib.save(saved_image, save_path)

        flow_filename = os.path.basename(moving_path)[:-4] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow[t].squeeze(), img=img[t].squeeze())



def inference_iterative_warp(path_list_gz, path_list_gt, newpath_flow, newpath_registered, model, image_size):
    motion_estimation = SpatialTransformer(size=(image_size, image_size)).to('cuda:0')

    patient_name = os.path.basename(path_list_gz[0]).split('_')[0]

    newpath_flow = os.path.join(newpath_flow, patient_name)
    newpath_registered = os.path.join(newpath_registered, patient_name)

    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    data = []
    for path in path_list_gz:
        arr, affine = vxm.py.utils.load_volfile(path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        x = torch.from_numpy(arr).to(device).float().permute(0, 4, 1, 2, 3)
        data.append(x)
    data = torch.stack(data, dim=0)
    data = data.permute(5, 0, 1, 2, 3, 4).contiguous()

    data_seg = []
    for path in path_list_gt:
        arr = vxm.py.utils.load_volfile(path, add_batch_axis=True, add_feat_axis=add_feat_axis)
        x = torch.from_numpy(arr).to(device).float().permute(0, 4, 1, 2, 3)
        data_seg.append(x)
    data_seg = torch.stack(data_seg, dim=0)
    data_seg = data_seg.permute(5, 0, 1, 2, 3, 4).contiguous()

    flow_list_all = []
    moved_list_all = []
    img_list_all = []
    for d in range(len(data)):
        current_data_depth_seg = data_seg[d]
        current_data_depth = data[d]

        out_list_flow_depth = []
        out_list_img = []

        for t in range(len(current_data_depth) - 1):
            x = torch.stack([current_data_depth[t], current_data_depth[t + 1]], dim=0)
            x = NormalizeIntensity()(x)

            out1, out2 = model(x, inference=False)
            warp = out2['flow'] # B, C, H, W

            out_list_flow_depth.append(warp)
            out_list_img.append(current_data_depth[t])
        
        flow = torch.stack(out_list_flow_depth, dim=0) # T, B, C, H, W
        img_sequence = torch.stack(out_list_img, dim=0) # T, B, C, H, W
        assert len(flow) == len(img_sequence) == len(path_list_gz) - 1

        moved_list = []
        current_moving_seg = current_data_depth_seg[-1]
        ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        for t in reversed(range(len(flow))):
            ed_target = motion_estimation(flow=flow[t], original=ed_target, mode='bilinear')
            moved = torch.argmax(ed_target, dim=1, keepdim=True).int() # B, 1, H, W
            moved_list.append(moved)
        moved = torch.stack(moved_list, dim=0)
        assert len(moved) == len(current_data_depth) - 1

        #moved_list = []
        #for t in range(len(flow)):
        #    current_moving_seg = current_data_depth_seg[t + 1]
        #    ed_target = torch.nn.functional.one_hot(current_moving_seg[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        #    for t2 in reversed(range(t + 1)):
        #        ed_target = motion_estimation(flow=flow[t2], original=ed_target, mode='bilinear')
        #    moved = torch.argmax(ed_target, dim=1, keepdim=True).int() # B, 1, H, W
        #    moved_list.append(moved)
        #moved = torch.stack(moved_list, dim=0)
        #assert len(moved) == len(current_data_depth) - 1

        flow_list_all.append(flow.cpu())
        moved_list_all.append(moved.cpu())
        img_list_all.append(img_sequence.cpu())
    
    moved = np.stack(moved_list_all, axis=-1) # T-1, 1, 1, H, W, D
    img = np.stack(img_list_all, axis=-1) # T-1, 1, 1, H, W, D
    flow = np.stack(flow_list_all, axis=-1) # T-1, 1, 2, H, W, D
    flow = flow.transpose(0, 1, 3, 4, 5, 2) # T-1, 1, H, W, D, 2

    for t in range(len(moved)):
        # save moved image
        moving_path = path_list_gz[t + 1]

        filename = os.path.basename(moving_path)
        save_path = os.path.join(newpath_registered, filename)
        vxm.py.utils.save_volfile(moved[t].squeeze(), save_path, affine)

        flow_filename = os.path.basename(moving_path)[:-7] + '.npz'
        save_path_flow = os.path.join(newpath_flow, flow_filename)
        np.savez(save_path_flow, flow=flow[t].squeeze(), img=img[t].squeeze())



# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--test_or_val', required=True, help='Whether this is testing set or validation_set')
parser.add_argument('--dataset', required=True, help='dataset (ACDC or Lib)')
parser.add_argument('--dirpath', required=True, help='output directory path')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

newpath_flow = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Backward_flow')
delete_if_exist(newpath_flow)
os.makedirs(newpath_flow)

newpath_registered = os.path.join(args.dirpath, args.dataset, args.test_or_val, 'Raw', 'Registered_backward')
delete_if_exist(newpath_registered)
os.makedirs(newpath_registered)

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

add_feat_axis = not args.multichannel

if args.dataset == 'ACDC':
    with open(os.path.join('splits', 'ACDC', 'splits_final.pkl'), 'rb') as f:
        data = pickle.load(f)
        training_patients = data[0]['train']
        validation_patients = data[0]['val']
    
    if args.test_or_val == 'test':
        path_list = glob(os.path.join('voxelmorph_ACDC_2D_testing', '*.gz'))
        path_list_gt = glob(os.path.join('voxelmorph_ACDC_gt_2D_testing', '*.gz'))
        path_list_pkl = glob(os.path.join('voxelmorph_ACDC_2D_testing', '*.pkl'))
    elif args.test_or_val == 'val':
        path_list = glob(os.path.join('voxelmorph_ACDC_2D', '*.gz'))
        path_list_gt = glob(os.path.join('voxelmorph_ACDC_gt_2D', '*.gz'))
        path_list_pkl = glob(os.path.join('voxelmorph_ACDC_2D', '*.pkl'))
    image_size = 128
elif args.dataset == 'Lib':
    if args.test_or_val == 'test':

        with open(os.path.join('splits', 'Lib', 'test', 'splits_final.pkl'), 'rb') as f:
            data = pickle.load(f)
            validation_patients = data[0]['val']

        path_list = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_testing_mask", '*.npy'))
        #path_list_pkl = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_testing_mask", '*.pkl'))

    elif args.test_or_val == 'val':

        with open(os.path.join('splits', 'Lib', 'val', 'splits_final.pkl'), 'rb') as f:
            data = pickle.load(f)
            validation_patients = data[0]['val']

        path_list = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\Lib_resampling_training_mask", '*.npy'))
    path_list_pkl = glob(os.path.join(r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\custom_lib_t_4", '**', '*.pkl'), recursive=True)
    image_size = 192

# load and set up model
# load the model

# load the model
model = Registration_Net()
model.load_state_dict(torch.load(args.model))
model = model.cuda()
model.eval()

validation_patients = sorted(list(set([x.split('_')[0] for x in validation_patients])))

path_list = sorted([x for x in path_list if os.path.basename(x).split('_')[0] in validation_patients])
path_list_pkl = sorted([x for x in path_list_pkl if os.path.basename(os.path.dirname(x)) in validation_patients])

assert len(path_list) == len(path_list_pkl)

patient_list = sorted(list(set([os.path.basename(x).split('_')[0] for x in path_list])))

all_patient_paths = []
all_patient_paths_pkl = []
for patient in patient_list:
    patient_files = []
    patient_files_pkl = []
    for (path, pkl_path) in zip(path_list, path_list_pkl):
        if patient in path:
            patient_files.append(path)
        if patient in pkl_path:
            patient_files_pkl.append(pkl_path)
    all_patient_paths.append(sorted(patient_files))
    all_patient_paths_pkl.append(sorted(patient_files_pkl))


for (path_list_gz, path_list_pkl) in tqdm(zip(all_patient_paths, all_patient_paths_pkl), total=len(all_patient_paths)):

    with open(path_list_pkl[0], 'rb') as f:
        data = pickle.load(f)
        ed_number = np.rint(data['ed_number']).astype(int)
        es_number = np.rint(data['es_number']).astype(int)

    path_list_gz = np.array(path_list_gz)
    frame_indices = np.arange(len(path_list_gz))
    after = frame_indices >= ed_number
    before = frame_indices < ed_number
    path_list_gz = np.concatenate([path_list_gz[after], path_list_gz[before]])

    assert int(os.path.basename(path_list_gz[0]).split('frame')[-1][:2]) == ed_number + 1

    with torch.no_grad():
        inference_iterative(path_list_gz, newpath_flow, newpath_registered, model, image_size)

    
