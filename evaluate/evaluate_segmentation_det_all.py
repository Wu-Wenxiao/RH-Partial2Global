import os.path
from tqdm import trange
import pascal_dataloader
from evaluate_detection.box_ops import to_rectangle
from evaluate_detection.canvas_ds import CanvasDataset
from reasoning_dataloader import *
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
from copy import deepcopy
import cv2


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='/shared/yossi_gandelsman/code/occlusionwalk/pascal', help='pascal base dir')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', help='model checkpoint')
    parser.add_argument('--dataset_type', default='pascal',
                        choices=['pascal', 'pascal_det'])
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--feature_name', default='features_rn50_val_det', type=str)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--annotation_file', default='', type=str)


    return parser


def _generate_result_for_ens(args, model, canvases, method='sum'):
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    num_patches = 14
    if method == 'sum':
        canvas, canvas2 = canvases[0], canvases[1]
        mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
        _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
        x1 = torch.softmax(x1, dim=-1)
        x2 = torch.softmax(x2, dim=-1)
        y = ((x1 + x2)/2).argmax(dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    elif method == 'sum_pre':
        canvas, canvas2 = canvases[0], canvases[1]
        mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
        _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
        y = ((x1 + x2) / 2).argmax(dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    elif method == 'mult':
        canvas, canvas2 = canvases[0], canvases[1]
        mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
        _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
        x1 = torch.softmax(x1, dim=-1)
        x2 = torch.softmax(x2, dim=-1)
        y = (x1 * x2).argmax(dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    elif method == 'max':
        canvas, canvas2 = canvases[0], canvases[1]
        mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
        _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
        x1 = torch.softmax(x1, dim=-1)
        x2 = torch.softmax(x2, dim=-1)
        y = torch.argmax(torch.max(torch.stack([x1,x2], dim=-1), dim=-1)[0], dim=-1)
        im_paste, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y)

    # elif method == 'union':
    #     canvas, canvas2 = canvases[0], canvases[1]
    #     mask, orig_image, x1 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas.unsqueeze(0))
    #     _, _, x2 = generate_raw_prediction(args.device, ids_shuffle, len_keep, model, canvas2.unsqueeze(0))
    #     y1 = torch.argmax(x1, dim=-1)
    #     y2 = torch.argmax(x2, dim=-1)
    #     im_paste1, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y1)
    #     im_paste2, mask, orig_image = decode_raw_predicion(mask, model, num_patches, orig_image, y2)
    #
    #
    else:
        raise ValueError("Wrong ens")
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()

    return np.uint8(canvas), np.uint8(im_paste[0]), mask


def _generate_result_for_canvas(args, model, canvas):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_mask_for_evaluation()
    _, im_paste, _ = generate_image(canvas.unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                    len_keep, device=args.device)
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip((canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)

def get_bbox(img):
    coords = np.where(img!=0)
    try:
        left, right = coords[1].min(), coords[1].max()
        up, bottom = coords[0].min(), coords[0].max()
        return (left, up, right, bottom)
    except:
        return (0,0,0,0)

def evaluate(args):
    with open(os.path.join(args.output_dir, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')
    with open(os.path.join(args.output_dir, 'log_name.txt'), 'w') as log:
        log.write(str(args) + '\n')
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    # import pdb;pdb.set_trace()
    ds = {
        'pascal': pascal_dataloader.DatasetPASCAL,
        'pascal_det': CanvasDataset
    }[args.dataset_type](args.base_dir, fold=args.split, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, random=args.random, cluster=args.cluster, feature_name=args.feature_name,
                         annotation_file=args.annotation_file)
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)
    # Build the transforms:
    eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
    all_data = []

    for idx in trange(len(ds)):
        ious = []

        generated_results = []
        for sim_idx in trange(0, 612):
            canvas = ds[(idx,sim_idx)]['grid_stack']
            s_name = ds[(idx,sim_idx)]['support_name']
            q_name = ds[(idx,sim_idx)]['query_name']
            # breakpoint()
            if args.dataset_type != 'pascal_det':
                canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            # Calculate the original_image and the result
            original_image, generated_result = _generate_result_for_canvas(args, model, canvas)

            # generated_results.append(generated_result)

        
        # generated_result = np.stack(generated_results, 0).mean(0)
            generated_result = np.int32(generated_result)
            original_image_ = deepcopy(original_image)
            original_image = round_image(original_image, [WHITE, BLACK])

            generated_result = round_image(generated_result, [WHITE, BLACK], t=args.t)

            if args.task == 'detection':
                generated_result = to_rectangle(generated_result)

            # import pdb;pdb.set_trace()

            current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)

            ious.append(current_metric['iou'])
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
                log.write(str(idx) + '\t' + str(sim_idx) + '\t' + str(current_metric) + '\n')
            with open(os.path.join(args.output_dir, 'log_name.txt'), 'a') as log:
                log.write(str(q_name) + '\t' + str(s_name) + '\t' + str(current_metric) + '\n')
            for i, j in current_metric.items():
                eval_dict[i] += (j / len(ds))

        # original_image_[:, 113:] = generated_result[:, 113:]

        # example_image = original_image_[:111, :111]
        # example_bbox = get_bbox(generated_result[:111, 113:])
        # query_image = original_image_[113:, :111]
        # query_bbox = get_bbox(generated_result[113:, 113:])

        # example_image = cv2.rectangle(np.uint8(np.ascontiguousarray(example_image)), (example_bbox[0], example_bbox[1]), (example_bbox[2], example_bbox[3]), (0, 255, 0), 2)
        # query_image = cv2.rectangle(np.uint8(np.ascontiguousarray(query_image)), (query_bbox[0], query_bbox[1]), (query_bbox[2], query_bbox[3]), (0, 255, 0), 2)

        # out_image = np.concatenate([example_image, query_image], 1)
        # all_data.append({"query":query_name, "gallery":support_name, "score":list(map(str, ious))})

        # Image.fromarray(np.uint8(out_image)).save(
        #     os.path.join(args.output_dir, '{}_{}.png'.format(ds[(idx,sim_idx)]['query_name'], current_metric['iou'])))
        

    # with open(os.path.join(args.output_dir, 'log.txt'), 'a') as log:
    #     log.write('all\t' + str(eval_dict) + '\n')
    print(str(eval_dict))

if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)

