import torch
from tensorboardX import SummaryWriter
import numpy as np

# mask = torch.randn([100, 64, 64])
# mask = torch.unsqueeze(mask, 1)
# print(mask.shape)
# print(mask.permute(0,2,3,1).shape)

# writer = SummaryWriter('runs/toy/')

# print(mask.shape[0])
# for i in range(mask[:5].shape[0]):
#       print(mask[i].shape)

# writer.add_images('b', mask, 0)
# writer.close()      

feature = torch.randn([184, 512])
b = [0,1,34]
print(feature[b].shape)