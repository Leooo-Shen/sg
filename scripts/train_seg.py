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
from sg2im.utils import timeit, bool_flag, LossManager

import clip
from tensorboardX import SummaryWriter

# torch.backends.cudnn.benchmark = True

VG_DIR = os.path.expanduser('datasets/vg')
COCO_DIR = os.path.expanduser('datasets/coco')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

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
parser.add_argument('--embedding_dim', default=512, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
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

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  curr_loss = curr_loss * weight
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss


def check_args(args):
  H, W = args.image_size
  for _ in args.refinement_network_dims[1:]:
    H = H // 2
  if H == 0:
    raise ValueError("Too many layers in refinement network")


def build_model(args, vocab):
  kwargs = {
      'vocab': vocab,
      'image_size': args.image_size,
      'embedding_dim': args.embedding_dim,
      'gconv_dim': args.gconv_dim,
      'gconv_hidden_dim': args.gconv_hidden_dim,
      'gconv_num_layers': args.gconv_num_layers,
      'mlp_normalization': args.mlp_normalization,
      'refinement_dims': args.refinement_network_dims,
      'normalization': args.normalization,
      'activation': args.activation,
      'mask_size': args.mask_size,
      'layout_noise_dim': args.layout_noise_dim,
    }
  model = Sg2ImModel(**kwargs)
  
  return model, kwargs


def build_obj_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_obj_weight = args.d_obj_weight
  if d_weight == 0 or d_obj_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'vocab': vocab,
    'arch': args.d_obj_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
    'object_size': args.crop_size,
  }
  discriminator = AcCropDiscriminator(**d_kwargs)
  return discriminator, d_kwargs


def build_img_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_img_weight = args.d_img_weight
  if d_weight == 0 or d_img_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'arch': args.d_img_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
  }
  discriminator = PatchDiscriminator(**d_kwargs)
  return discriminator, d_kwargs


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
    'shuffle': True,
    'collate_fn': collate_fn,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)
  
  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader


def check_model(args, t, loader, model):
  # float_dtype = torch.cuda.FloatTensor
  # long_dtype = torch.cuda.LongTensor
  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 7:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1] 

      # Run the model as it has been run during training
      masks_pred = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
      out = F.binary_cross_entropy(masks_pred, masks.float())

      # skip_pixel_loss = False
      # total_loss, losses =  calculate_model_losses(
      #                           args, skip_pixel_loss, model, imgs, imgs_pred,
      #                           boxes, boxes_pred, masks, masks_pred,
      #                           predicates, predicate_scores)

      # total_iou += jaccard(boxes_pred, boxes)
      # total_boxes += boxes_pred.size(0)

      # for loss_name, loss_val in losses.items():
      #   all_losses[loss_name].append(loss_val)
      # num_samples += imgs.size(0)
      # if num_samples >= args.num_val_samples:
      #   break

    # samples = {}
    # samples['gt_img'] = imgs

    # model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
    # samples['gt_box_gt_mask'] = model_out[0]

    # model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
    # samples['gt_box_pred_mask'] = model_out[0]

    # model_out = model(objs, triples, obj_to_img)
    # samples['pred_box_pred_mask'] = model_out[0]

    # for k, v in samples.items():
    #   samples[k] = imagenet_deprocess_batch(v)

    # mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    # avg_iou = total_iou / total_boxes

    # masks_to_store = masks
    # if masks_to_store is not None:
    #   masks_to_store = masks_to_store.data.cpu().clone()

    # masks_pred_to_store = masks_pred
    # if masks_pred_to_store is not None:
    #   masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

  # batch_data = {
  #   'objs': objs.detach().cpu().clone(),
  #   'boxes_gt': boxes.detach().cpu().clone(), 
  #   'masks_gt': masks_to_store,
  #   'triples': triples.detach().cpu().clone(),
  #   'obj_to_img': obj_to_img.detach().cpu().clone(),
  #   'triple_to_img': triple_to_img.detach().cpu().clone(),
  #   'boxes_pred': boxes_pred.detach().cpu().clone(),
  #   'masks_pred': masks_pred_to_store
  # }
  # out = [mean_losses, samples, batch_data, avg_iou]
  # out = [mean_losses, avg_iou]

  return out


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
  
  
def main(args):
  # print(args)
  if args.debug:
    args.checkpoint_every = 1
    args.print_every = 1
    args.batch_size = 4
    
  device = "cuda" if torch.cuda.is_available() else "cpu"
  writer = SummaryWriter("runs/sg2im_clip")
  
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  
  check_args(args)
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor

  vocab, train_loader, val_loader = build_loaders(args)
  
  print("Using ", torch.cuda.device_count(), " GPUs!")
  if torch.cuda.device_count() > 1:
    ids = [0,1]
  else:
    ids = [0]
    
  model, model_kwargs = build_model(args, vocab)
  model.type(float_dtype)
  # print(model)  ## Sg2ImModel
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
  
  ## Dataparallel
  model = nn.DataParallel(model, device_ids=ids)
  clip_model, _ = clip.load("ViT-B/32", device=device)
  clip_model.eval()
  
  for param in clip_model.parameters():
    param.requires_grad = False
  clip_model = nn.DataParallel(clip_model, device_ids=ids, download_root='./pretrained_weights')
  
  

  ## training
  t, epoch = 0, 0
  while True:
    if t >= args.num_iterations:
      break
    epoch += 1
    print('Starting epoch %d' % epoch)
    
    for batch in train_loader:
      
      # if t == args.eval_mode_after:
      #   print('switching to eval mode')
      #   model.eval()
        
      t += 1
      batch = [tensor.cuda() for tensor in batch]
      

      imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1]

      ## model forward
      with timeit('forward', args.timing):
        model_boxes = boxes
        model_masks = masks 
        prompt = []
        
        # print(vocab["object_idx_to_name"])
        for obj_idx in objs.cpu().detach().numpy():
          obj_name = vocab["object_idx_to_name"][obj_idx]
          prompt.append(create_prompt(obj_name))
          
        text = clip.tokenize(prompt).to(objs.device)
        with torch.no_grad():
          clip_features = clip_model.module.encode_text(text)
          
        masks_pred = model(objs, triples, obj_to_img,
                          boxes_gt=model_boxes, masks_gt=model_masks, clip_features=clip_features)
        
        print(masks_pred.shape, model_masks.shape)
        ## get losses
        with timeit('loss', args.timing):
          loss = F.binary_cross_entropy(masks_pred, masks.float())
      _masks_pred = masks_pred.detach().cpu().numpy()
      _masks = masks.detach().cpu().numpy()
      print(np.histogram(_masks_pred))
      print(np.histogram(_masks))
      
      print(loss)
      if not math.isfinite(loss.item()):
        print('WARNING: Got loss = NaN, not backpropping')
        continue

      ## backward
      optimizer.zero_grad()
      
      with timeit('backward', args.timing):
        loss.backward()
      optimizer.step()
      
      ## output printings
      if t % args.print_every == 0:
        print('t = %d / %d, loss: %.4f' % (t, args.num_iterations, loss.item()))

        writer.add_scalar('loss', loss, t)
        writer.add_images('masks_pred', masks_pred, t) 
        writer.add_images('masks', masks, t) 
        
      ## save checkpoints and print
      if t % args.checkpoint_every == 0:
        # print('checking on train')
        # t_bi_crs_entropy = check_model(args, t, train_loader, model)
        # print('t_loss: %.4f '%  t_bi_crs_entropy.item())
        # print('t_loss: %.4f '%  loss.item())
        
        print('checking on val')
        val_bi_crs_entropy = check_model(args, t, val_loader, model)
        print('val_loss: %.4f '%  val_bi_crs_entropy)
        
        writer.add_scalar('train_loss', loss, t)
        writer.add_scalar('val_loss', val_bi_crs_entropy, t)


        checkpoint_path = os.path.join(args.output_dir,'%s_%s_it%d_loss%.4f.pt' 
                                       % (args.checkpoint_name, args.dataset, t, loss.item()))
        print('saving checkpoints to:', checkpoint_path)
        torch.save(model.module.state_dict(), checkpoint_path)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

