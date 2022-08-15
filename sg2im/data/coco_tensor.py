# a dataset class to load saved tensor data
import os
import torch
from torch.utils.data import Dataset


class TransformedDataset(Dataset):
  def __init__(self, batch_dir):
    self.batch_dir = batch_dir
    self.len = len(os.listdir(batch_dir))

  def __getitem__(self, index):
    ls_batch = os.listdir(self.batch_dir)

    batch_file_path = os.path.join(self.batch_dir, ls_batch[index])
    batch_tensor = torch.load(batch_file_path)

    # imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch_tensor
    # return imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img
    return batch_tensor

  def __len__(self):
    return self.len   

def coco_collate_fn2(batch):
  """
  Collate function to be used when wrapping CocoSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving object categories
  - boxes: FloatTensor of shape (O, 4)
  - masks: FloatTensor of shape (O, M, M)
  - triples: LongTensor of shape (T, 3) giving triples
  - obj_to_img: LongTensor of shape (O,) mapping objects to images
  - triple_to_img: LongTensor of shape (T,) mapping triples to images
  """
  all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0

  for i, (img, objs, boxes, masks, triples, obj_to_img, triple_to_img) in enumerate(batch):
    img = img.squeeze(0)
    # print(img.shape, objs.shape, boxes.shape, masks.shape, triples.shape, obj_to_img.shape, triple_to_img.shape)

    all_imgs.append(img[None])
    if objs.dim() == 0 or triples.dim() == 0:
      continue
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    all_masks.append(masks)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_masks = torch.cat(all_masks)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
         all_obj_to_img, all_triple_to_img)
  return out