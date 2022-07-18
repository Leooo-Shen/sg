import torch
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn as nn

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

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# print(input.shape, target.shape)
# # loss = nn.CrossEntropyLoss()
# loss = nn.L1Loss()
# output = loss(input, target)

import segmentation_models_pytorch as pytorch_models

unet3 = pytorch_models.Unet(
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    encoder_depth=3,                # Amount of down- and upsampling of the Unet
    decoder_channels=(64, 32,16),   # Amount of channels
    encoder_weights = None,         # Model does not download pretrained weights
    # activation = 'softmax'           # Activation function to apply after final convolution       
    )

t = torch.randn([8,1,64,64])
out = unet3(t)
print(out.shape)


# model = pytorch_models.FPN('resnet18', 
#                            in_channels=1,
#                            encoder_weights = None,
#                            classes=184,)
# mask = model(torch.ones([1, 1, 64, 64]))
# print(mask.shape)

# thres = torch.LongTensor([0.9])
# print(type(thres))

# t = torch.tensor([ 1,   1,   1,  96, 105, 132, 140, 173,   0,   1,   1,  96, 120, 140,
#         144,   0,   2,   8,  96, 140, 149, 169,   0,  70, 117, 123, 143, 176,
#         181,   0,   4, 102, 105, 115, 130, 173,   0,  70,  81, 105, 117, 123,
#         132, 152, 173,   0,  64,  64, 112, 116, 126, 172,   0,  51, 123, 172,
#           0])
# print(len(t))

# mask = torch.randn([57, 64, 64])
# idx = np.array([i.item() for i in (t==0).nonzero()]) + 1
# diff = list(np.diff(idx))
# diff.insert(0, idx[0])

# masks_split = torch.split(mask, diff)
# t_split = torch.split(t, diff)

# for i in range(len(masks_split)):
#       print(masks_split[i].shape, t_split[i].shape)
#       print((masks_split[i] * t_split[i].unsqueeze(dim=-1). unsqueeze(dim=-1)).shape)


mask = torch.FloatTensor([[              
              [0, 0.2, 1, 0, 0],
              [0, 0.9, 0.95, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 0.3, 1, 1, 0],
              [0.2, 0, 1, 0, 0]],
                          
              [[0, 0.2, 1, 0, 0],
              [0, 0.9, 0.95, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 0.3, 1, 1, 0],
              [0.2, 0, 1, 0, 0],]
                          ])

# print(mask.shape)
out = (mask>0.5).float()
# print(out)

t = torch.tensor([2,3])
res = out * t.unsqueeze(dim=-1). unsqueeze(dim=-1)
print(res)
