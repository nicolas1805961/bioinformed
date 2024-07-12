import torch
import numpy as np
import os
import pickle
from scipy.ndimage import zoom
import cv2
from kornia.morphology import dilation, erosion

def compute_gradient(x):
    # compute gradients of deformation fields x =[u, v]
    # x: deformation field with 2 channels as x- and y- dimensional displacements
    # du/dx = (u(x+1)-u(x-1)/2
    bsize, csize, height, width = x.size()
    xw = torch.cat((torch.zeros(bsize, csize, height, 1).cuda(), x, torch.zeros(bsize, csize, height, 1).cuda()), 3)
    d_x = (torch.index_select(xw, 3, torch.arange(2, width+2).cuda()) - torch.index_select(xw, 3, torch.arange(width).cuda()))/2  #[du/dx, dv/dx]
    xh = torch.cat((torch.zeros(bsize, csize, 1, width).cuda(), x, torch.zeros(bsize, csize, 1, width).cuda()), 2)
    d_y = (torch.index_select(xh, 2, torch.arange(2, height+2).cuda()) - torch.index_select(xh, 2, torch.arange(height).cuda()))/2  #[du/dy, dv/dy]
    d_xy = torch.cat((d_x, d_y), 1)
    d_xy = torch.index_select(d_xy, 1, torch.tensor([0, 2, 1, 3]).cuda()) #[du/dx, du/dy, dv/dx, dv/dy]
    return d_xy

def compute_strain(x):
    # compute gradients of deformation fields x =[u, v]
    # x: deformation field with 2 channels as x- and y- dimensional displacements
    # du/dx = (u(x+1)-u(x-1)/2
    bsize, csize, height, width = x.size()
    xw = torch.cat((torch.zeros(bsize, csize, height, 1).cuda(), x, torch.zeros(bsize, csize, height, 1).cuda()), 3)
    d_x = (torch.index_select(xw, 3, torch.arange(2, width+2).cuda()) - torch.index_select(xw, 3, torch.arange(width).cuda()))/2  #[du/dx, dv/dx]
    xh = torch.cat((torch.zeros(bsize, csize, 1, width).cuda(), x, torch.zeros(bsize, csize, 1, width).cuda()), 2)
    d_y = (torch.index_select(xh, 2, torch.arange(2, height+2).cuda()) - torch.index_select(xh, 2, torch.arange(height).cuda()))/2  #[du/dy, dv/dy]
    d_xy = torch.cat((d_x, d_y), 1)
    d_xy = torch.index_select(d_xy, 1, torch.tensor([0, 2, 1, 3]).cuda()) #[du/dx, du/dy, dv/dx, dv/dy]

    dudx = d_xy[:,0,:,:].view(bsize, 1, height, width)
    dudy = d_xy[:,1,:,:].view(bsize, 1, height, width)
    dvdx = d_xy[:,2,:,:].view(bsize, 1, height, width)
    dvdy = d_xy[:,3,:,:].view(bsize, 1, height, width)

    # print(dudx.shape.view(bsize, 1, height, width))
    strain_tensor = torch.cat((dudx, dvdy, 1/2*(dudy+dvdx)), 1)

    return strain_tensor

def centre_crop(img, size, centre):
    h1 = max(0, centre[0] - size//2)
    h2 = min(centre[0] + size//2, img.shape[1])
    x1 = max(0, centre[1] - size//2)
    x2 = min(centre[1] + size//2, img.shape[2])

    cropped = img[:, h1:h2, x1:x2]

    to_crop_data = (0, size, 0, size)
    to_pad_data = (x1, max(0, img.shape[2] - x2), h1, max(0, img.shape[1] - h2))

    if list(cropped.shape[1:]) != [size, size]:
        pad_top = max(0, size//2 - (centre[0]))
        pad_bottom = max(0, size//2 - (img.shape[1] - centre[0]))
        pad_left = max(0, size//2 - (centre[1]))
        pad_right = max(0, size//2 - (img.shape[2] - centre[1]))
        cropped = np.pad(cropped, pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
        to_crop_data = (pad_top, size - pad_bottom, pad_left, size - pad_right)

    return cropped, to_crop_data, to_pad_data


def func_loadTestResults(model_name):
    mov_list, fix_list, moved_list, dvf_list, mov_seg_list, fix_seg_list= [], [], [], [], [] ,[]

    result_path = './results'
    result_names_list = os.listdir(os.path.join(result_path, model_name))
    for result_name in result_names_list:
        result_load_path = os.path.join(result_path, model_name, result_name)
        with open(result_load_path, 'rb') as f:
            prediction = pickle.load(f)
        
        mov = prediction['mov']
        fix = prediction['fix']
        moved = prediction['pred_moved']
        dvf = prediction['pred_dvf']
        mov_seg = prediction['mov_seg']
        fix_seg = prediction['fix_seg']

        # print(mov.shape)
        # print(fix.shape)
        # print(moved.shape)
        # print(dvf.shape)
        # print(mov_seg.shape)
        # print(fix_seg.shape)

        mov_list.append(mov[0,0,:,:]), fix_list.append(fix[0,0,:,:]), moved_list.append(moved[0,0,:,:]), dvf_list.append(dvf[0,:,:,:]), mov_seg_list.append(mov_seg[0,0,:,:]), fix_seg_list.append(fix_seg[0,0,:,:])
    
    return mov_list, fix_list, moved_list, dvf_list, mov_seg_list, fix_seg_list


def func_normPixSpacing(img, target_spacing = np.array([1.25, 1.25, 8])):
    # img is in raw nifty format by default

    img_header = img.header
    img_spacing = np.array([img_header['pixdim'][1], img_header['pixdim'][2], img_header['pixdim'][3]])
    new_img = zoom(img.get_data(), img_spacing/target_spacing)

    return new_img


