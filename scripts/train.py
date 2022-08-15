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

# from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.data.coco_tensor import TransformedDataset, coco_collate_fn2
import matplotlib.pyplot as plt

import clip
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

VG_DIR = os.path.expanduser('datasets/vg')
# COCO_DIR = os.path.expanduser('datasets/coco')
COCO_DIR = '/root/autodl-tmp/datasets/coco'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=128, type=int)
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
parser.add_argument('--loader_num_workers', default=12, type=int)
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
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
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
parser.add_argument('--print_every', default=50, type=int)
parser.add_argument('--checkpoint_every', default=1000, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--output_dir', default='/root/autodl-tmp/checkpoints/SG2IM_CLIP')
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
    'shuffle': False,
    'collate_fn': collate_fn,
    'pin_memory': False,
    'drop_last': True
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)
  
  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader


def check_model(args, t, loader, model, clip_features):
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
      clip_embeddings = clip_features[objs]
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks, clip_features=clip_embeddings)
      imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out

      skip_pixel_loss = False
      total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs, imgs_pred,
                                boxes, boxes_pred, masks, masks_pred,
                                predicates, predicate_scores)

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

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

    mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    avg_iou = total_iou / total_boxes

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
  out = [mean_losses, avg_iou, imgs, imgs_pred]

  return tuple(out)


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
  if args.debug:
    args.checkpoint_every = 100
    args.print_every = 5
    # args.batch_size = 8
    args.coco_train_image_dir = args.coco_val_image_dir
    args.coco_train_instances_json = args.coco_val_instances_json
    args.coco_train_stuff_json = args.coco_val_stuff_json
    
  # print(args)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  writer = SummaryWriter("/root/tf-logs/")
  # writer = SummaryWriter("runs")

  
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  
  check_args(args)
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor

  # vocab, train_loader, val_loader = build_loaders(args)
  ## save data as tensor
  # if not os.path.exists('/root/autodl-tmp/tensors'):
  #   os.makedirs('/root/autodl-tmp/tensors/val')
  #   os.makedirs('/root/autodl-tmp/tensors/train')

  # print('saving transformed tensor data...')
  # for i, batch in enumerate(train_loader):
  #   torch.save(batch, '/root/autodl-tmp/tensors/train/transformed{}'.format(i))
  #   print('%d / %d' %(i, len(train_loader)))

  # for i, batch in enumerate(val_loader):
  #   print(len(batch))
  #   # torch.save(batch, '/root/autodl-tmp/tensors/val/transformed{}'.format(i))
  #   print('%d / %d' %(i, len(val_loader)))
  # exit()

  vocab, _, _ = build_coco_dsets(args)
  train_tensor = TransformedDataset('/root/autodl-tmp/tensors/train')
  val_tensor = TransformedDataset('/root/autodl-tmp/tensors/val')
  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': False,
    'collate_fn': coco_collate_fn2,
    'pin_memory': True,
    'drop_last': True,
  }
  train_loader = DataLoader(train_tensor, **loader_kwargs)
  val_loader = DataLoader(val_tensor, **loader_kwargs)
  print("Tensor dataset len: ", len(train_loader))
  print("Using ", torch.cuda.device_count(), " GPUs!")
    
  model, model_kwargs = build_model(args, vocab)
  model.to(device)
  # print(model)  ## Sg2ImModel
  obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, vocab)
  img_discriminator, d_img_kwargs = build_img_discriminator(args, vocab)
  obj_discriminator.to(device)
  img_discriminator.to(device)

  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
  gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)
  
  ## clip features
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

  
  if obj_discriminator is not None:
    # obj_discriminator.type(float_dtype)
    obj_discriminator.train()
    # print(obj_discriminator)
    optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(),
                                       lr=args.learning_rate)

  if img_discriminator is not None:
    # img_discriminator.type(float_dtype)
    img_discriminator.train()
    # print(img_discriminator)
    optimizer_d_img = torch.optim.Adam(img_discriminator.parameters(),
                                       lr=args.learning_rate)

  scaler = GradScaler()
  ckp_path = '/root/autodl-tmp/checkpoints/SG2IM_CLIP/sg2im_clip_coco_it110000_loss0.6102_viou0.4843.pt'
  state_dict = torch.load(ckp_path)
  model.load_state_dict(state_dict)
  print('[*] Successfully load model: ', ckp_path)

  ## training
  t, epoch = 110000, 0
  while True:
    if t >= args.num_iterations:
      break
    epoch += 1
    print('Starting epoch %d' % epoch)
    
    for batch in train_loader:
      t += 1
      batch = [tensor.cuda() for tensor in batch]
      
      ## for some settings, we could use GT masks
      # masks = None
      # if len(batch) == 6:
      #   imgs, objs, boxes, triples, obj_to_img, _ = batch
      # elif len(batch) == 7:
      imgs, objs, boxes, masks, triples, obj_to_img, _ = batch
      # else:
      #   assert False
      predicates = triples[:, 1]

      # imgs = imgs.squeeze(0)
      # print(imgs.shape, objs.shape, boxes.shape, masks.shape, 
      # triples.shape, obj_to_img.shape)

      ## save images debug
      # print(de_imgs.shape)
      # for i in range(de_imgs.shape[0]):
      #   plt.imsave('./images/%d.png'%i, de_imgs[i].permute(1,2,0).cpu().numpy())
      # exit()
      ## model forward
      with autocast():
        model_boxes = boxes
        model_masks = masks 
        prompt = []

        # print(vocab["object_idx_to_name"])
          
        clip_embeddings = clip_features[objs]
        model_out = model(objs, triples, obj_to_img,
                          boxes_gt=model_boxes, masks_gt=model_masks, clip_features=clip_embeddings)
        imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      
        # Skip the pixel loss if using GT boxes
        skip_pixel_loss = (model_boxes is None)
        total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs, imgs_pred,
                                boxes, boxes_pred, masks, masks_pred,
                                predicates, predicate_scores)

        ## object discriminator loss
        if obj_discriminator is not None:
          scores_fake, ac_loss = obj_discriminator(imgs_pred, objs, boxes, obj_to_img)
          total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss',
                                args.ac_loss_weight)
          weight = args.discriminator_loss_weight * args.d_obj_weight
          total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                'g_gan_obj_loss', weight)

        ## image discriminator loss
        if img_discriminator is not None:
          scores_fake = img_discriminator(imgs_pred)
          weight = args.discriminator_loss_weight * args.d_img_weight
          total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                'g_gan_img_loss', weight)

      losses['total_loss'] = total_loss.item()
      if not math.isfinite(losses['total_loss']):
        print('WARNING: Got loss = NaN, not backpropping')
        continue

      ## backward
      optimizer.zero_grad()
      scaler.scale(total_loss).backward()
      scaler.step(optimizer)
      scaler.update()
      # total_loss.backward()
      # optimizer.step()
      
      total_loss_d = None
      ac_loss_real = None
      ac_loss_fake = None
      d_losses = {}
      
      ## train object discriminator
      if obj_discriminator is not None:
        with autocast():
          d_obj_losses = LossManager()
          imgs_fake = imgs_pred.detach()
          scores_fake, ac_loss_fake = obj_discriminator(imgs_fake, objs, boxes, obj_to_img)
          scores_real, ac_loss_real = obj_discriminator(imgs, objs, boxes, obj_to_img)

          d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
          d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
          d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
          d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')
        # d_obj_losses.total_loss.backward()
        # optimizer_d_obj.step()
        optimizer_d_obj.zero_grad()
        scaler.scale(d_obj_losses.total_loss).backward()
        scaler.step(optimizer_d_obj)
        scaler.update()

      ## train image discriminator
      if img_discriminator is not None:
        with autocast():
          d_img_losses = LossManager()
          imgs_fake = imgs_pred.detach()
          # print(11, imgs.shape, imgs_fake.shape)
          scores_fake = img_discriminator(imgs_fake)
          scores_real = img_discriminator(imgs)
          d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
          d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')
        # d_img_losses.total_loss.backward()
        # optimizer_d_img.step()
      
        optimizer_d_img.zero_grad()
        scaler.scale(d_img_losses.total_loss).backward()
        scaler.step(optimizer_d_img)
        scaler.update()
        
      ## output printings
      if t % args.print_every == 0:
        print('t = %d / %d' % (t, args.num_iterations))

        de_imgs = imagenet_deprocess_batch(imgs.float())
        de_pred = imagenet_deprocess_batch(imgs_pred.float())
        writer.add_images('imgs_gt', de_imgs, t)
        writer.add_images('imgs_pred', de_pred, t) 
        
        for name, val in losses.items():
          print(' G [%s]: %.4f' % (name, val))
          writer.add_scalar('G '+ name, val, t)
        # writer.add_scalars('G_loss', losses, t)

        if obj_discriminator is not None:
          for name, val in d_obj_losses.items():
            print(' D_obj [%s]: %.4f' % (name, val))
            writer.add_scalar('D_obj '+ name, val, t)
            # checkpoint['d_losses'][name].append(val)
          # writer.add_scalars('D_obj_loss', d_obj_losses, t)
          
        if img_discriminator is not None:
          for name, val in d_img_losses.items():
            print(' D_img [%s]: %.4f' % (name, val))
            writer.add_scalar('D_img '+ name, val, t)
            # checkpoint['d_losses'][name].append(val)
          # writer.add_scalars('D_img_loss', d_img_losses, t)
          
      ## save checkpoints and print
      if t % args.checkpoint_every == 0:
        # print('checking on train')
        # t_losses, t_avg_iou = check_model(args, t, train_loader, model)
        # print('train_iou: %.4f '%  t_avg_iou)
        
        print('checking on val')
        model.eval()
        
        val_losses, val_avg_iou, val_imgs, val_imgs_pred = check_model(args, t, val_loader, model, clip_features)
        for name, val in val_losses.items():
          print(' Val [%s]: %.4f' % (name, val))
          writer.add_scalar('val' + name, val, t)
            
        print(' Val_iou: %.4f '%  val_avg_iou)   
        de_val_imgs = imagenet_deprocess_batch(val_imgs.float())
        de_val_pred = imagenet_deprocess_batch(val_imgs_pred.float())
        writer.add_images('val_imgs_gt', de_val_imgs, t)
        writer.add_images('val_imgs_pred', de_val_pred, t) 
        # writer.add_scalars('val_avg_loss', val_losses, t)
        writer.add_scalar('val_iou', val_avg_iou, t)

        checkpoint_path = os.path.join(args.output_dir,'%s_%s_it%d_loss%.4f_viou%.4f.pt' 
                                       % (args.checkpoint_name, args.dataset, t, losses['total_loss'], val_avg_iou))
        print('saving checkpoints to:', checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        
        model.train()


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

