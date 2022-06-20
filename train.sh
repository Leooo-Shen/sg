#!/bin/bash

python scripts/train.py --dataset=vg --image_size=64,64

# python scripts/train.py --dataset=coco --image_size=64,64 --checkpoint_start_from ./checkpoints/SG2IM_CLIP/sg2im_clip_no_model_1_viou_0.021.pt