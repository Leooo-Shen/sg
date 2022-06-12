
import torch
import clip
from PIL import Image
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/workspace/sg2im/CLIP/2.jpg")).unsqueeze(0).to(device)
print(image.shape)

text = clip.tokenize(["a picture of a diagram", "a dog", "a cat", "a cow"]).to(device)  # torch.Size([3, 77])

with torch.no_grad():
    image_features = model.encode_image(image)  # [I_batch, 512]
    text_features = model.encode_text(text)   #   [T_batch, 512]
    
    out = image_features @ text_features.T   # [I_batch, T_batch]
    print(F.softmax(out, dim=-1))            # highest num (nearly 1.0) for the class
    
    
    
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]