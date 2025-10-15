import os.path
import torchvision
import torch
from tqdm import trange
import pascal_dataloader_all
from evaluate_detection.box_ops import to_rectangle
from evaluate_detection.canvas_ds import CanvasDataset
from reasoning_dataloader import *
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 2010_000866':
# {'2007_001420': 0.020078741
query_name = "2007_000032"
support_name = "2007_004459"
dataset_type = "pascal"
base_dir = "./dataset"
split = "trn"
feature_name = "features_vit-laion2b-in21k_trn"
padding = 1

def _generate_result_for_canvas(model, canvas):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, _ = generate_image(canvas.unsqueeze(0).cuda(), model, ids_shuffle.cuda(),
                                    len_keep, device='cuda')
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)
image_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
        torchvision.transforms.ToTensor()])
mask_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
        torchvision.transforms.ToTensor()])
ckpt_path = "./checkpoint-1000.pth"
arch = "mae_vit_large_patch16"
model = prepare_model(ckpt_path, arch=arch)
_ = model.cuda()
# model = model.to(args.device)
# Build the transforms:
ds = pascal_dataloader_all.DatasetPASCAL(base_dir, fold=0, split=split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=0, purple=0, random=False, cluster=False, feature_name=feature_name, percentage='', seed=0)


canvas = ds.get_grid(query_name, support_name)[0]

print(canvas.shape)
# breakpoint()
if dataset_type != 'pascal_det':
    canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

original_image, generated_result = _generate_result_for_canvas(model, canvas)
# generated_results = []
# generated_results.append(generated_result)
# generated_result = np.stack(generated_results, 0).mean(0)
original_image = round_image(original_image, [WHITE, BLACK])
generated_result_1 = round_image(np.int32(generated_result), [WHITE, BLACK], t=[0, 0, 0])
generated_result_2 = round_image(generated_result, [WHITE, BLACK], t=[0, 0, 0])
# Image.fromarray(np.uint8(generated_result)).save(
#                         os.path.join("./", f'1.png'))
args = None
current_metric_1 = calculate_metric(args, original_image, generated_result_1, fg_color=WHITE, bg_color=BLACK)
current_metric_2 = calculate_metric(args, original_image, generated_result_2, fg_color=WHITE, bg_color=BLACK)
print(current_metric_2)
print(current_metric_1)