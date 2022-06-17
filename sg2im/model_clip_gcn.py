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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sg2im.box_utils as box_utils
from sg2im.graph import GraphTripleConv, GraphTripleConvNet
from sg2im.crn import RefinementNetwork
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.layers import build_mlp

import clip

class Sg2ImModel(nn.Module):
  def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
               gconv_dim=128, gconv_hidden_dim=512,
               gconv_pooling='avg', gconv_num_layers=5,
               refinement_dims=(1024, 512, 256, 128, 64),
               normalization='batch', activation='leakyrelu-0.2',
               mask_size=None, mlp_normalization='none', layout_noise_dim=0,
               batch_size=8,project_dim=None,
               **kwargs):
    super(Sg2ImModel, self).__init__()

    # We used to have some additional arguments: 
    # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
    if len(kwargs) > 0:
      print('WARNING: Model got unexpected kwargs ', kwargs)

    self.vocab = vocab
    self.image_size = image_size
    self.layout_noise_dim = layout_noise_dim
    self.batch_size = batch_size
    # print(self.vocab.keys()) 
    # dict_keys(['object_name_to_idx', 'object_idx_to_name', 'attribute_name_to_idx', 'attribute_idx_to_name', 'pred_name_to_idx', 'pred_idx_to_name'])
    
    num_objs = len(vocab['object_idx_to_name'])
    num_preds = len(vocab['pred_idx_to_name'])
    
    ## embedding layers
    self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)
  

    ## construct GCN
    if gconv_num_layers == 0:
      self.gconv = nn.Linear(embedding_dim, gconv_dim)
    elif gconv_num_layers > 0:
      gconv_kwargs = {
        'input_dim': embedding_dim,
        'output_dim': gconv_dim,
        'hidden_dim': gconv_hidden_dim,
        'pooling': gconv_pooling,
        'mlp_normalization': mlp_normalization,
      }
      self.gconv = GraphTripleConv(**gconv_kwargs)

    self.gconv_net = None
    if gconv_num_layers > 1:
      gconv_kwargs = {
        'input_dim': gconv_dim,
        'hidden_dim': gconv_hidden_dim,
        'pooling': gconv_pooling,
        'num_layers': gconv_num_layers - 1,
        'mlp_normalization': mlp_normalization,
      }
      self.gconv_net = GraphTripleConvNet(**gconv_kwargs)


    # proj_kwargs = {
    #     'input_dim': embedding_dim,
    #     'output_dim': 1,
    #     'hidden_dim': gconv_hidden_dim,
    #     'pooling': gconv_pooling,
    #     'mlp_normalization': mlp_normalization,
    #   }
    # self.project = GraphTripleConv(**proj_kwargs)

  def forward(self, objs, triples, obj_to_img=None,
              clip_features=None):
    """
    Required Inputs:
    - objs: LongTensor of shape (O,) giving categories for all objects
    - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
      means that there is a triple (objs[s], p, objs[o])

    Optional Inputs:
    - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
      means that objects[o] is an object in image i. If not given then
      all objects are assumed to belong to the same image.
    - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
      the spatial layout; if not given then use predicted boxes.
    """

    
    O, T = objs.size(0), triples.size(0)
    s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
    s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
    edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)
  
    if obj_to_img is None:
      obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)


    ## get object embeddings
    if clip_features is None:
      obj_vecs = self.obj_embeddings(objs)  # torch.Size([len(objs), 512])
    else:
      obj_vecs = clip_features.float()
    pred_vecs = self.pred_embeddings(p)
    

    ## GCN calculation
    if isinstance(self.gconv, nn.Linear):
      obj_vecs = self.gconv(obj_vecs)
    else:
      obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
    if self.gconv_net is not None:
      obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

    # obj_vecs dim=[obj, 512]
    obj_vecs = obj_vecs.mean(dim=0)
    
    return obj_vecs  
  
    

class GraphCLIP(nn.Module):
  def __init__(self,vocab, temperature=1.0, **kwargs):
        super().__init__()
        
        self.temperature = temperature
        self.gcn = Sg2ImModel(vocab, **kwargs)
        self.clip_model, _ = clip.load("ViT-B/32")
        self.vocab = vocab
      
      
  def create_prompt(self, obj_name):
    assert isinstance(obj_name, str)
    return "A photo of a " + obj_name
  
  
  def forward(self, imgs, objs, triples, obj_to_img=None):
    prompt = []
    self.clip_model.eval()
    
    # use CLIP to get object_prompt features
    for obj_idx in objs.cpu().detach().numpy():
      obj_name = self.vocab["object_idx_to_name"][obj_idx]
      prompt.append(self.create_prompt(obj_name))
    
    with torch.no_grad():
      image_embeddings = self.clip_model.encode_image(imgs)
      text = clip.tokenize(prompt).to(image_embeddings.device)
      # text = clip.tokenize(prompt)
      text_embeddings = self.clip_model.encode_text(text)
  
    gcn_embeddings = self.gcn(objs, triples, obj_to_img,clip_features=text_embeddings)
    
    # Calculating the Loss
    logits = (gcn_embeddings @ image_embeddings.T) / self.temperature
    images_similarity = image_embeddings @ image_embeddings.T
    gcn_similarity = gcn_embeddings @ gcn_embeddings.T
    targets = F.softmax(
        (images_similarity + gcn_similarity) / 2 * self.temperature, dim=-1
    )
    texts_loss = self.cross_entropy(logits, targets)
    images_loss = self.cross_entropy(logits.T, targets.T)
    loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    return loss.mean()
  
    
  def cross_entropy(self, preds, targets, reduction='none'):
      log_softmax = nn.LogSoftmax(dim=-1)
      loss = (-targets * log_softmax(preds)).sum(1)
      if reduction == "none":
          return loss
      elif reduction == "mean":
          return loss.mean()

  