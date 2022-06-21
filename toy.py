import torch
from tensorboardX import SummaryWriter

mask = torch.randn([100, 64, 64])
# mask = torch.unsqueeze(mask, 1)
print(mask.unsqueeze(1).shape)

# writer = SummaryWriter('runs/toy/')

# print(mask.shape[0])
# for i in range(mask[:5].shape[0]):
#       print(mask[i].shape)

# writer.add_images('b', mask, 0)
# writer.close()      
