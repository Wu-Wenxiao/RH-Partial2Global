"""
Extract features for UnsupPR.
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm, trange
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F

import timm
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


model_name = sys.argv[1]
feature_name = sys.argv[2]
split = sys.argv[3]

"""
model names:
# vit_large_patch14_224_clip_laion2b
# eva_large_patch14_196.in22k_ft_in22k_in1k
# resnet50
# vit_large_patch16_224.augreg_in21k_ft_in1k
# resnet18
# vit_large_patch14_clip_224.laion2b_ft_in12k_in1k
# vit_large_patch14_clip_224.laion2b_ft_in12k
# vit_base_patch16_224.dino
"""
# model = timm.create_model(model_name, pretrained=True)
print(model_name)
model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./weights/pytorch_model.bin'))
print(model.default_cfg)
model.eval()
model = model.cuda()
# breakpoint()


# load the image transformer
t = []
t.append(T.Resize(model.pretrained_cfg['input_size'][1], interpolation=Image.BICUBIC))
t.append(T.CenterCrop(model.pretrained_cfg['input_size'][1]))
t.append(T.ToTensor())
t.append(T.Normalize(model.pretrained_cfg['mean'], model.pretrained_cfg['std']))
center_crop = T.Compose(t)

# t = []
# size = 224
# t.append(T.Resize((size,size), interpolation=Image.BICUBIC))
# t.append(T.CenterCrop(size))
# t.append(T.ToTensor())
# t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
# center_crop = T.Compose(t)


save_dir = f"./dataset/VOC2012/{feature_name}_{split}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    sys.exit()


meta_root =  f"./evaluate/splits/pascal/{split}"
image_root = "./dataset/VOC2012/JPEGImages"
for folder_id in [0,1,2,3]:
    print(f"Processing folder {folder_id}")
    sys.stdout.flush()
    with open(os.path.join(meta_root, 'fold'+str(folder_id)+'.txt')) as f:
        examples = f.readlines()
    if len(examples) == 0:
        print(f"zeros folder{folder_id}")
        sys.stdout.flush()
        continue

    examples = [os.path.join(image_root, example.strip()[:-4]+'.jpg') for example in examples]
        
    imgs = []

    global_features = torch.tensor([]).cuda()
    for example in examples:
        try:
            path = os.path.join(example)
            img = Image.open(path).convert("RGB")
            img = center_crop(img)
            imgs.append(img)
        except:
            print(f"Disappear {path}")
            sys.stdout.flush()

        if len(imgs) == 128:

            imgs = torch.stack(imgs).cuda()
            with torch.no_grad():
                features = model.forward_features(imgs)
                # breakpoint()
                features = model.forward_head(features,pre_logits=True)
                if len(global_features) == 0:
                    global_features = features
                else:
                    global_features = torch.cat((global_features,features))

            imgs = []

    imgs = torch.stack(imgs).cuda()
    with torch.no_grad():
        features = model.forward_features(imgs)
        features = model.forward_head(features,pre_logits=True)
        if len(global_features) == 0:
            global_features = features
        else:
            global_features = torch.cat((global_features,features))

    features = global_features.cpu().numpy().astype(np.float32)
    
    save_file = os.path.join(save_dir, 'folder'+str(folder_id))
    np.savez(save_file, examples=examples, features=features)
