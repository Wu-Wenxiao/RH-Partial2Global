from losses import listMLE, approxNDCGLoss, neuralNDCG, MarginLoss
from model import RankModel
from dataset import RankTestDataset
import os
CUDA_ID = os.environ.get('CUDA_ID')
os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_ID)
import torch
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import json
from copy import deepcopy
import math
import argparse
import random
import itertools
from hodge_utils import hodge_rank
import time

import numpy as np


# seed=43
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# seed = int(np.floor(np.datetime64('now') / np.timedelta64(1, 's')))
# np.random.seed(seed)
# print("numpy seed: {} | random seed {} | torch seed {}".format(np.random.get_state()[1][0], random.getstate()[1][0], torch.initial_seed()))
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--source_fold_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_gallery_list",
        nargs='+',
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='dinov1',
    )
    parser.add_argument(
        "--similarity_annotation",
        type=str,
        default='dinov1',
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default='./ckpt',
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default='/mnt/tmp/JPEGImages',
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default='jpg',
    )
    parser.add_argument(
        "--num",
        type=int,
        default=850,
    )
    # parser.add_argument(
    #     "--least_num",
    #     type=int,
    #     default=20,
    # )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )



    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    return args

        

def get_edge_set(pred, idx_set):
    edge_set = []
    sorted_idx = pred.argsort(descending=True)
    for comb in itertools.combinations(range(len(idx_set)), 2):
        edge_set.append((idx_set[sorted_idx[comb[0]]], idx_set[sorted_idx[comb[1]]]))

    return edge_set

def get_edge_set_by_n(model, query_img, gallery_img, gallery_idx, n_gallery, only_random=False):
    edge_set = []

    if not only_random:
        # print(2)
        gallery_img_ = gallery_img.clone()
        gallery_idx_ = deepcopy(gallery_idx)
        # sequential select

        # start = time.time()
        while gallery_img_.shape[1] > 1:
            all_candidate = []
            all_candidate_idx = []

            for j in range(int(math.ceil(gallery_img_.shape[1]/n_gallery))):
                with torch.no_grad(), torch.autocast("cuda"):
                    pred = model(query_img, gallery_img_[:, j*n_gallery:(j+1)*n_gallery])
                pred = pred.view(-1)
                preg_argsort = pred.argsort()
                # print(time.time() - start)

                all_candidate.append(gallery_img_[:, j*n_gallery+preg_argsort[-1].item()])
                all_candidate_idx.append(gallery_idx_[j*n_gallery+preg_argsort[-1].item()])

                # start = time.time()
                edge_set.extend(get_edge_set(pred, gallery_idx_[j*n_gallery:(j+1)*n_gallery]))

            
            gallery_img_ = torch.stack(all_candidate, dim=1)
            gallery_idx_ = deepcopy(all_candidate_idx)

        # print('Sequential gallery:', time.time() - start)

    # start = time.time()
    # random select
    n_random = 50 if not only_random else 70

    if len(gallery_idx) <= n_gallery:
        for _ in range(n_random):
            rand_idx = random.sample(range(len(gallery_idx)), len(gallery_idx))
            # print(rand_idx)
            gallery_img_ = gallery_img[:, rand_idx]
            gallery_idx_ = [gallery_idx[j] for j in rand_idx]
            with torch.no_grad(), torch.autocast("cuda"):
                pred = model(query_img, gallery_img_).view(-1)
            edge_set.extend(get_edge_set(pred, gallery_idx_))
            # breakpoint()

    else:
        with open("./vkt_cd/v_{}-k_{}-t_{}.json".format(len(gallery_idx), n_gallery,2)) as f:
            sample_list = json.load(f)
            n_sample = len(sample_list)
            all_n = n_random//n_sample + 1

        for _ in range(all_n):
        # gallery_idx 保留一开始的顺序，真实idx
        # rand_idx 打乱后的顺序，乱序的idx
        # sample_idx 按照固定的采样列表从打乱后的顺序中取出，为对应的真实idx: rand_idx[j] for j in sample
            rand_idx = list(range(len(gallery_idx)))
            random.shuffle(rand_idx)
            gallery_img_ = gallery_img[:, rand_idx]
            # breakpoint()
            for sample in sample_list:
                sample_idx = [rand_idx[j-1] for j in sample]
                gallery_img_ = gallery_img[:, sample_idx]
                gallery_idx_ = [gallery_idx[k] for k in sample_idx]
                assert sample_idx == gallery_idx_
                with torch.no_grad(), torch.autocast("cuda"):
                    pred = model(query_img, gallery_img_).view(-1)
                edge_set.extend(get_edge_set(pred, gallery_idx_))

    return edge_set

args = parse_args()

fold_idx = args.fold_idx
num_layers = 3
num_heads = 8
if 'clip-l' in args.backbone:
    hidden_dim = mlp_dim  = 1024
else:
    hidden_dim = mlp_dim = 768
n_patch = 4
n_init_gallery = 50

train_batch_size = 64
dataloader_num_workers = 8

num_epoch = 30
lr = 1e-4
weight_decay = 0

model = RankModel(num_layers, num_heads, hidden_dim, mlp_dim, n_patch, backbone=args.backbone)
model.backbone.requires_grad_(False)
model.backbone.eval()
model = model.cuda()
model.eval()

cudnn.benchmark = True

state_dict_list = []
n_gallery_list = list(map(lambda x:int(x), args.n_gallery_list))
for n_gallery in n_gallery_list:
    ckpt = torch.load('./ckpt_xu/all/dinov2_fold{}_rank{}.pth'.format(args.source_fold_idx, n_gallery))
    new_state_dict = {}
    for k, v in ckpt.items():
        new_state_dict[k.replace('module.', '')] = v
    # msg = model.load_state_dict(new_state_dict, strict=False)
    state_dict_list.append(new_state_dict)


root = args.data_root
train_transforms  =  transforms.Compose([
    transforms.CenterCrop(size=224),
    # transforms.RandomApply([
    #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    # ], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
print(args.similarity_annotation)
try:
    train_dataset = RankTestDataset(root, args.similarity_annotation, 
                                train_transforms,
                                n_gallery=50,
                                n_init_gallery=1,
                                image_suffix=args.image_suffix)
except:
    train_dataset = RankTestDataset(root, args.similarity_annotation, 
                                train_transforms,
                                n_gallery=50,
                                n_init_gallery=1,
                                image_suffix=args.image_suffix)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=dataloader_num_workers,
    )

# filter_result = []

filter_result = {}
all_models = []
for state_dict_idx, n_gallery in enumerate(n_gallery_list):
    model_ = deepcopy(model)
    model_.load_state_dict(state_dict_list[state_dict_idx], strict=False)
    all_models.append(model_)

rm_path = "./topall_json/kl_set_fold{}_{}_sum1.json".format(args.source_fold_idx, args.alpha)

print(f"cp-rm-path: {rm_path}")

with open(rm_path) as f:
    kl_set = json.load(f)
    # kl_set = []

# progress_bar = tqdm(range(len(train_loader)))
# least = args.least_num
# print("least num: {}".format(least))
for i, (query_img, gallery_img, query_name, gallery_name) in enumerate(tqdm(train_loader)):
    # print(query_name[0])
    # print(gallery_img.shape)
    save_index = []
    remove_index = []
    remove_name = []
    save_name = []
    gallery_name_copy = deepcopy(gallery_name)
    for i in range(len(gallery_name_copy)):
        # print(i)
        if gallery_name_copy[i][0] not in kl_set:
            save_index.append(i)
            save_name.append(gallery_name_copy[i][0])
        else:
            gallery_name.remove(gallery_name_copy[i])
            remove_index.append(i)
            remove_name.append(gallery_name_copy[i][0])

    
    # if len(save_index) < least:

    #     add_len = least-len(save_index)
        
    #     rm_cor = []
    #     for name in remove_name:
    #         rm_cor.append(cor_map[name])
    #     rm_cor_arr = np.array(rm_cor)

    #     add_index = rm_cor_arr.argsort()[-add_len:]

    #     for a_index in add_index:
    #         save_index.append(remove_index[a_index])
    #         gallery_name.append((remove_name[a_index],0))
    #         # breakpoint()
    #         assert gallery_name_copy[remove_index[a_index]][0] == remove_name[a_index]


    #     assert len(save_index)==least
        
    

    # breakpoint()
    gallery_img = gallery_img[:,save_index]
    # assert len(save_index)==50
    # breakpoint()
    # gallery_name = gallery_name[save_index]
    # assert len(gallery_name)==gallery_img.shape[1]
    # print(gallery_img.shape[1])

    query_img, gallery_img = query_img.to('cuda', non_blocking=True), \
            gallery_img.to('cuda', non_blocking=True),
    gallery_idx = list(range(gallery_img.shape[1]))

    edge_set = []

    for state_dict_idx, n_gallery in enumerate(n_gallery_list):
        # print(state_dict_idx)
        msg = model.load_state_dict(state_dict_list[state_dict_idx], strict=False)
        # print(msg)
        edge_set_ = get_edge_set_by_n(all_models[state_dict_idx],query_img, gallery_img, gallery_idx, n_gallery, only_random=False)
        edge_set.extend(edge_set_)

    start = time.time()
    hodge = hodge_rank(edge_set, len(gallery_idx))
    best_idx = hodge.get_global_rank().argsort()[-5:]
    # breakpoint()
    # print(time.time() - start)
    # print(len(gallery_name))
    output_gallery = [gallery_name[idx][0] for idx in best_idx]
    output_gallery.reverse()
    # print(output_gallery[0])

    # filter_result.append({"query":query_name[0], "gallery":output_gallery})
    filter_result[query_name[0]] = output_gallery
    # progress_bar.update(1)


w = open('./output_json_for_re_rh/sourcefold{}_fold{}_alpha{}_rank{}_{}_hodge_{}.json'.format(args.source_fold_idx, fold_idx, args.alpha, '_'.join(args.n_gallery_list), args.backbone, args.suffix), 'w')
json.dump(filter_result, w, indent=4)
