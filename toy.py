import torch

mask = torch.randn([100, 64, 64])

print(mask.shape[0])
for i in range(mask[:5].shape[0]):
      print(mask[i].shape)
      
      