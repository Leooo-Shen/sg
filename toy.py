import torch
from tensorboardX import SummaryWriter

mask = torch.randn([100, 1, 64, 64])


writer = SummaryWriter('runs/toy/')

print(mask.shape[0])
for i in range(mask[:5].shape[0]):
      print(mask[i].shape)

writer.add_images('b', mask, 0)
writer.close()      
