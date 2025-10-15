import json
import os
import sys
from tqdm import tqdm


inference_result_root = './output_seg_images/'

feature_name = sys.argv[1]
# feature_name = "features_vit-laion2b-in12k_trn"
# output_name = "output_vit-laion2b_trn-all"
output_name = sys.argv[2]

for fold_id in [0,1,2,3]:
    cur_dir = os.path.join(inference_result_root, f'all_{output_name}_folder{fold_id}_seed0/log_all.txt')
    
    with open(cur_dir) as f:
        metas = f.readlines()
        print(len(metas))
    
    fold_n_metadata_path = f'./evaluate/splits/pascal/trn/fold{fold_id}.txt'

    with open(fold_n_metadata_path, 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
        
    # import pdb;pdb.set_trace()
    fold_n_metadata = [data.split('__')[0] for data in fold_n_metadata]

    with open(f'./dataset/VOC2012/{feature_name}/folder{fold_id}_all-similarity.json') as f:
        images_top50 = json.load(f)

    iou_dict = {}
    cur_dict = {}
    cur_dict['query'] = ""
    cur_dict['gallery'] = []
    cur_dict['score'] = []
    cur_save_dict = {}
    for k, cur_line in enumerate(tqdm(metas[1:-1])):
        # import pdb;pdb.set_trace()
        img_id, sim_id, result = cur_line.split('\t')
        img_id, sim_id = int(img_id), int(sim_id)
        result = eval(result)
        iou = result['iou']
        image_name = fold_n_metadata[img_id]
        gallery_name = images_top50[image_name][sim_id]
        if image_name == gallery_name:
            continue
        if image_name != cur_dict['query']:
            if cur_dict['gallery'] != []:
                # iou_dict.append(cur_dict)
                iou_dict[cur_dict['query']] = cur_save_dict
                # if k > 114200:
                #     print(cur_dict['query'])
                #     print(cur_save_dict)
                #     print(cur_dict['query'])
                #     breakpoint()

            cur_dict = {}
            cur_save_dict = {}
            cur_dict['query'] = image_name
            cur_dict['gallery'] = []
            cur_dict['score'] = []
            cur_dict['gallery'].append(gallery_name)
            cur_dict['score'].append(iou)
            cur_save_dict[gallery_name] = iou
        elif image_name == cur_dict['query']:
            cur_dict['gallery'].append(gallery_name)
            cur_dict['score'].append(iou)
            cur_save_dict[gallery_name] = iou
    iou_dict[cur_dict['query']] = cur_save_dict
    print(cur_dict['query'])
    print(len(iou_dict))
    save_dir = os.path.join(inference_result_root, f'all_{output_name}_folder{fold_id}_seed0/annotation-all.json')
    with open(save_dir,'w') as f:
        json.dump(iou_dict, f)




