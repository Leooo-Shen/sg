#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from asyncore import write
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco_seg import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard
from sg2im.model_seg import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, dice_loss
import segmentation_models_pytorch as pytorch_models

import clip
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

VG_DIR = os.path.expanduser('datasets/vg')
# COCO_DIR = os.path.expanduser('datasets/coco')
COCO_DIR = os.path.expanduser('/mnt/nfs-datasets-students/coco')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco', 'coco_debug'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
# TODO: modify here for other generation options 
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=8, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=64, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight 
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--d_img_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight

# Output options
parser.add_argument('--print_every', default=500, type=int)
parser.add_argument('--checkpoint_every', default=50000, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--output_dir', default='checkpoints/SG2IM_CLIP')
parser.add_argument('--checkpoint_name', default='sg2im_clip')
parser.add_argument('--restore_from_checkpoint', default=False, type=bool)

parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--overfit', default=False, type=bool)


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  curr_loss = curr_loss * weight
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss


# def check_args(args):
#   H, W = args.image_size
#   for _ in args.refinement_network_dims[1:]:
#     H = H // 2
#   if H == 0:
#     raise ValueError("Too many layers in refinement network")


def build_model(args, vocab):
  kwargs = {
      'vocab': vocab,
      'image_size': args.image_size,
      'embedding_dim': args.embedding_dim,
      'gconv_dim': args.gconv_dim,
      'gconv_hidden_dim': args.gconv_hidden_dim,
      'gconv_num_layers': args.gconv_num_layers,
      'mlp_normalization': args.mlp_normalization,
      'normalization': args.normalization,
      'activation': args.activation,
      'mask_size': args.mask_size,
      'layout_noise_dim': args.layout_noise_dim,
    }
  model = Sg2ImModel(**kwargs)
  
  return model, kwargs



def build_coco_dsets(args):
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
    'stuff_only': args.coco_stuff_only,
    'image_size': args.image_size,
    'mask_size': args.mask_size,
    'max_samples': args.num_train_samples,
    'min_object_size': args.min_object_size,
    'min_objects_per_image': args.min_objects_per_image,
    'instance_whitelist': args.instance_whitelist,
    'stuff_whitelist': args.stuff_whitelist,
    'include_other': args.coco_include_other,
    'include_relationships': args.include_relationships,
  }
  train_dset = CocoSceneGraphDataset(**dset_kwargs)
  num_objs = train_dset.total_objects()
  num_imgs = len(train_dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  dset_kwargs['image_dir'] = args.coco_val_image_dir
  dset_kwargs['instances_json'] = args.coco_val_instances_json
  dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  dset_kwargs['max_samples'] = args.num_val_samples
  val_dset = CocoSceneGraphDataset(**dset_kwargs)

  assert train_dset.vocab == val_dset.vocab
  vocab = json.loads(json.dumps(train_dset.vocab))

  return vocab, train_dset, val_dset


def build_vg_dsets(args):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = VgSceneGraphDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = VgSceneGraphDataset(**dset_kwargs)
  
  return vocab, train_dset, val_dset


def build_loaders(args):
  if args.dataset == 'vg':
    vocab, train_dset, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn
  elif args.dataset == 'coco':
    vocab, train_dset, val_dset = build_coco_dsets(args)
    collate_fn = coco_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': False,
    'collate_fn': collate_fn,
    'pin_memory': False,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)
  
  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader


def check_model(loader, model, clip_features):
  model.eval()
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      objs, masks, boxes, triples, obj_to_img, seg_maps= batch
      clip_embeddings = clip_features[objs]
      
      boxes_pred, masks_pred = model(objs, triples, obj_to_img=obj_to_img, 
                            boxes_gt=boxes, clip_features=clip_embeddings)
      simple_sum = sum_masks(masks_pred, objs) # [batch, 1, 64, 64]      

      loss_mask = F.binary_cross_entropy(masks_pred, masks.float())
      loss_bbox = F.mse_loss(boxes_pred, boxes)
      criterian = nn.L1Loss()
      loss_seg = criterian(simple_sum, seg_maps)
      total_loss = loss_mask  + loss_bbox + loss_seg * 0.01
  return (total_loss, loss_mask, loss_bbox, loss_seg)


def calculate_model_losses(args, skip_pixel_loss, model, img, img_pred,
                           bbox, bbox_pred, masks, masks_pred,
                           predicates, predicate_scores):
  total_loss = torch.zeros(1).to(img)
  losses = {}

  l1_pixel_weight = args.l1_pixel_loss_weight
  if skip_pixel_loss:
    l1_pixel_weight = 0
  l1_pixel_loss = F.l1_loss(img_pred, img)
  total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                        l1_pixel_weight)
  loss_bbox = F.mse_loss(bbox_pred, bbox)
  total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        args.bbox_pred_loss_weight)

  if args.predicate_pred_loss_weight > 0:
    loss_predicate = F.cross_entropy(predicate_scores, predicates)
    total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                          args.predicate_pred_loss_weight)

  if args.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
    mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
                          args.mask_loss_weight)
  return total_loss, losses


def create_prompt(obj_name):
  assert isinstance(obj_name, str)
  return "A photo of a " + obj_name


def sum_masks(masks, objs):
  '''
  Input: concated object-wise masks
  Return: [batch, M, M] segmentation masks
  '''
  #   ## asign object class to each mask and sum them up
  # _masks = masks_pred.clone().detach()
  # for obj_cls, _mask in zip(objs, _masks):
  #   _mask[_mask>thres] = obj_cls.float()
  # mask_sum = masks_pred.sum(dim=0, keepdim=True)
  
  idx = np.array([i.item() for i in (objs==0).nonzero()]) + 1
  diff = list(np.diff(idx))
  diff.insert(0, idx[0])
  
  obj_split = torch.split(objs, diff)
  mask_split = torch.split(masks, diff)
  

  mask_sum = []
  for m, obj in zip(mask_split, obj_split):
    binary_m = (m>0.5).float()   # binaralize
    mask_cls = binary_m * obj.unsqueeze(dim=-1).unsqueeze(dim=-1)
    mask_sum.append(mask_cls.sum(dim=0, keepdim=True))
  mask_sum = torch.cat(mask_sum, dim=0)
  return mask_sum


def main(args):
  # print(args)
  if args.debug:
    args.checkpoint_every = 5
    args.print_every = 5
    args.batch_size = 8
    args.coco_train_image_dir = args.coco_val_image_dir
    args.coco_train_instances_json = args.coco_val_instances_json
    args.coco_train_stuff_json = args.coco_val_stuff_json
    
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print('Using', device)
  print("Using ", torch.cuda.device_count(), " GPUs!")
  writer = SummaryWriter()
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


  # if torch.cuda.device_count() > 1:
  #   ids = [0,1]
  #   # ids = [0]
  # else:
  #   ids = [0]
  
  vocab, train_loader, val_loader = build_loaders(args)
  overfit_batch = next(iter(train_loader))

  model, _ = build_model(args, vocab)
  model.to(device)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
  
  ## Dataparallel
  # model = nn.DataParallel(model, device_ids=ids)
  clip_model, _ = clip.load("ViT-B/32", device='cuda:0', download_root='./pretrained_weights')
  clip_model.eval()
  
  prompt = []
  for obj_name in vocab["object_idx_to_name"]:
    prompt.append(create_prompt(obj_name))
  text = clip.tokenize(prompt).to(device)
  
  with torch.no_grad():
    print('[*] Generating object features with CLIP...')
    clip_features = clip_model.encode_text(text).float()
  print('[*] Features successfully generated for %d objects' % clip_features.shape[0])
  
  ## training
  t, epoch = 0, 0
  while True:
    if t >= args.num_iterations:
      break
    epoch += 1
    print('Starting epoch %d' % epoch)
   
    if not args.overfit: 
      for batch in train_loader:
        t += 1
        batch = [tensor.to(device) for tensor in batch]
        objs, masks, boxes, triples, obj_to_img, seg_maps= batch

        with timeit('forward', args.timing):
          clip_embeddings = clip_features[objs]
          boxes_pred, masks_pred = model(objs, triples, obj_to_img=obj_to_img, 
                            boxes_gt=boxes, clip_features=clip_embeddings)
          
          simple_sum = sum_masks(masks_pred, objs) # [batch, 64, 64]      
          print(simple_sum.shape, seg_maps.shape)
          
          loss_mask = F.binary_cross_entropy(masks_pred, masks.float())
          loss_bbox = F.mse_loss(boxes_pred, boxes)
          criterian = nn.L1Loss()
          loss_seg = criterian(simple_sum, seg_maps)
          loss = loss_mask  + loss_bbox + loss_seg * 0.1
            
        if not math.isfinite(loss.item()):
          print('WARNING: Got loss = NaN, not backpropping')
          continue
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % args.print_every == 0:
          print('t = %d / %d, loss: %.4f' % (t, args.num_iterations, loss.item()))

          writer.add_scalar('train_loss_total', loss, t)
          writer.add_scalar('train_loss_mask', loss_mask, t)
          writer.add_scalar('train_loss_bbox', loss_bbox, t)
          writer.add_scalar('train_loss_seg', loss_seg, t)
          
          writer.add_images('masks_pred', masks_pred.unsqueeze(1), t) 
          writer.add_images('masks', masks.unsqueeze(1), t) 
          writer.add_images('simple_sum', simple_sum.unsqueeze(1), t) 
          writer.add_images('seg_gt', seg_maps.unsqueeze(1), t) 
          
        # save checkpoints 
        if t % args.checkpoint_every == 1:
          print('checking on val')
          loss, loss_mask, loss_bbox, loss_seg = check_model(val_loader, model, clip_features)
          print('val_loss: %.4f '%  loss)
          
          writer.add_scalar('val_loss_total', loss, t)
          writer.add_scalar('val_loss_mask', loss_mask, t)
          writer.add_scalar('val_loss_bbox', loss_bbox, t)
          writer.add_scalar('val_loss_seg', loss_seg, t)

          checkpoint_path = os.path.join(args.output_dir,'%s_%s_it%d_loss%.4f.pt' 
                                        % (args.checkpoint_name, args.dataset, t, loss.item()))
          print('saving checkpoints to:', checkpoint_path)
          # torch.save(model.module.state_dict(), checkpoint_path)
          torch.save(model.state_dict(), checkpoint_path)
    
    ## overfit mode      
    else: 
      unet3 = pytorch_models.Unet(
          in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
          classes=1,                      # model output channels (number of classes in your dataset)
          encoder_depth=3,                # Amount of down- and upsampling of the Unet
          decoder_channels=(64, 32,16),   # Amount of channels
          encoder_weights = None,         # Model does not download pretrained weights
          # activation = 'softmax'           # Activation function to apply after final convolution       
          )
    
      unet3.to(device)
      overfit_batch = [tensor.to(device) for tensor in overfit_batch]
      objs, masks, boxes, triples, obj_to_img, seg_maps = overfit_batch
      for t in range(1000000):
        clip_embeddings = clip_features[objs]
        boxes_pred, masks_pred = model(objs, triples, obj_to_img=obj_to_img, 
                          boxes_gt=boxes, clip_features=clip_embeddings)

        simple_sum = sum_masks(masks_pred, objs) # [batch, 1, 64, 64]

        # seg_pred = unet3(simple_sum) # [batch, 1, 64, 64]
        
        loss_mask = F.binary_cross_entropy(masks_pred, masks.float())
        loss_bbox = F.mse_loss(boxes_pred, boxes)
        criterian = nn.L1Loss()
        loss_seg = criterian(simple_sum, seg_maps)
        loss = loss_mask  + loss_bbox + loss_seg * 0.1
            
        if not math.isfinite(loss.item()):
          print('WARNING: Got loss = NaN, not backpropping')
          continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer2.step()
        
        if t % 5 == 0:
          print('t = %d / %d, loss: %.4f' % (t, args.num_iterations, loss.item()))
          writer.add_scalar('train_loss_total', loss, t)
          writer.add_scalar('train_loss_mask', loss_mask, t)
          writer.add_scalar('train_loss_bbox', loss_bbox, t)
          writer.add_scalar('train_loss_seg', loss_seg, t)
                            
          writer.add_images('masks_pred', masks_pred.unsqueeze(1), t) 
          writer.add_images('masks', masks.unsqueeze(1), t) 
          writer.add_images('simple_sum', simple_sum.unsqueeze(1), t) 
          writer.add_images('seg_gt', seg_maps.unsqueeze(1), t) 
          

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

