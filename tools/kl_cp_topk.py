from scipy.special import kl_div, softmax
import numpy as np
import json
import os
import sys

def scale(x):
    EPS = 0.000001
    return (x-x.min())/(x.max()-x.min())+EPS


# alpha = float(sys.argv[1])
# fold = int(sys.argv[2])
# for alpha in [0.05,0.1,0.15,0.2,0.25,0.3]
# for alpha in [0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
# for alpha in [0.06,0.07,0.08,0.09,0.11,0.12,0.13,0.14,0.16,0.17,0.18,0.19]:
for alpha in [0.05,0.1,0.15,0.2,0.25,0.3,0.06,0.07,0.08,0.09,0.11,0.12,0.13,0.14,0.16,0.17,0.18,0.19,0.21,0.22,0.23,0.24]:
    for fold in [0,1,2,3]:
        print("alpha {} | fold {}".format(alpha, fold))

        topk = None
        if fold == 0:
            annotation_path = "./output_seg_images/all_output_vit-laion2b_trn-all_folder0_seed0/annotation-all-float.json"
            sim_path = "./dataset/VOC2012/features_vit-laion2b-in21k_trn/folder0_similarity.json"

        if fold == 1:
            annotation_path = "./output_seg_images/all_output_vit-laion2b_trn-all_folder1_seed0/annotation-all.json"
            sim_path = "./dataset/VOC2012/features_vit-laion2b-in21k_trn/folder1_similarity.json"

        if fold == 2:
            annotation_path = "./output_seg_images/all_output_vit-laion2b_trn-all_folder2_seed0/annotation-all.json"
            sim_path = "./dataset/VOC2012/features_vit-laion2b-in21k_trn/folder2_similarity.json"

        if fold == 3:
            annotation_path = "./output_seg_images/output_vit-laion2b-in21k_trn_all_folder3_seed0/annotation-all-review.json"
            sim_path = "./dataset/VOC2012/features_vit-laion2b-in21k_trn/folder3_similarity.json"

        with open(annotation_path) as f:
            ious_ = json.load(f)
            print(len(ious_))

        with open(sim_path) as f:
            sims_ = json.load(f)
            print(len(sims_))

        kl_cp = {}
        kl_name = []
        kl_score = []
        # img_name 作为prompt
        for img_name in ious_.keys():
            sim_ = []
            iou_ = []
            for query_name in ious_[img_name].keys():
                if query_name == img_name:
                    continue
                else:
                    sim = sims_[img_name][query_name]
                    # assert sims_[img_name][query_name] == sims_[query_name][img_name]
                    if query_name not in ious_:
                        print(query_name)
                        continue
                    if img_name not in ious_[query_name]:
                        print(img_name)
                    else:
                        iou = ious_[query_name][img_name]
                        iou_.append(iou)
                        sim_.append(sim)
            if len(iou_) != 0:
                iou_arr = np.asarray(iou_)
                sim_arr = np.asarray(sim_)

                if topk is not None:

                    top_sim_index = sim_arr.argsort()[-50:]
                    sim_arr_top = sim_arr[top_sim_index]
                    iou_arr_top = iou_arr[top_sim_index]

                    iou_arr_top = scale(iou_arr_top)
                    iou_arr_top = iou_arr_top/(iou_arr_top.sum())
                    sim_arr_top = scale(sim_arr_top)
                    sim_arr_top = sim_arr_top/(sim_arr_top.sum())
                    assert len(iou_arr_top)==len(sim_arr_top)
                    # print(len(iou_arr))
                    kl = kl_div(iou_arr_top, sim_arr_top).sum()
                else:
                    
                    iou_arr = scale(iou_arr)
                    iou_arr = iou_arr/(iou_arr.sum())
                    sim_arr = scale(sim_arr)
                    sim_arr = sim_arr/(sim_arr.sum())
                    assert len(iou_arr)==len(sim_arr)
                    # print(len(iou_arr))
                    # print(len(iou_arr))
                    kl = kl_div(iou_arr, sim_arr).sum()
                    # print(kl)
                # kl = kl_div(sim_arr, iou_arr).sum()
                kl_cp[img_name] = kl
                kl_score.append(kl)
                kl_name.append(img_name)
        # print(len(kl_cp))
        # w = open("./test_1.json", "w")
        # json.dump(kl_cp, w)

        arry = np.array(kl_score)
        print(arry.mean())

        quantile_num = int(alpha*len(arry))
        print("fold {} | all_num: {} | quantile_num: {}".format(fold, len(arry), quantile_num))
        index = np.argsort(arry)[-quantile_num:]
        # print(arry[index[0]])
        kl_set = []
        for i in index:
            kl_set.append(kl_name[i])
        # print(kl_set)
        w = open("./topall_json/kl_set_fold{}_{}_sum1.json".format(fold, alpha), "w")
        # w = open("./topk_json/kl_set_fold{}_{}_sum1.json".format(fold, alpha), "w")
        json.dump(kl_set, w)
