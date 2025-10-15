import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json


features_name = sys.argv[1]
source_split = sys.argv[2]
target_split = sys.argv[3]

print(f"Processing {features_name} ...")
sys.stdout.flush()

source_features_dir = f"./dataset/VOC2012/{features_name}_{source_split}"
target_features_dir = f"./dataset/VOC2012/{features_name}_{target_split}"
print(source_features_dir)
print(target_features_dir)

for alpha in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
    for foldid in [0,1,2,3]:
        with open("./topall_json/kl_set_fold{}_{}_sum1.json".format(foldid,alpha)) as f:
            kl_set = json.load(f)
        feature_file = 'folder'+str(foldid)+'.npz'
        print(f"Processing {feature_file} ...")
        sys.stdout.flush()
        source_path = os.path.join(source_features_dir, feature_file)
        target_path = os.path.join(target_features_dir, feature_file)
        print(source_path)
        try:
            source_file_npz = np.load(source_path)
            target_file_npz = np.load(target_path)
        except:
            print(f"no folder {feature_file} ...")
            sys.stdout.flush()
            continue
        source_examples = source_file_npz["examples"].tolist()
        target_examples = target_file_npz["examples"].tolist()
        source_features = source_file_npz["features"].astype(np.float32)
        target_features = target_file_npz["features"].astype(np.float32)
        print(source_features.shape)
        print(target_features.shape)

        # breakpoint()

        target_sample_idx = np.random.choice(target_features.shape[0], size=int(target_features.shape[0]), replace=False)
        target_sample_feature = target_features[target_sample_idx,:]
        similarity = dot(source_features,target_sample_feature.T)/(linalg.norm(source_features,axis=1, keepdims=True) * linalg.norm(target_sample_feature,axis=1, keepdims=True).T)

        similarity_idx = np.argsort(similarity,axis=1)[:,-1500:]

        similarity_idx_dict = {}
        for i, (cur_example, cur_similarity) in enumerate(zip(source_examples,similarity_idx)):
            # print(i)
            img_name = cur_example.strip().split('/')[-1][:-4]

            save_all = []
            
            count = 0
            for idx in cur_similarity[::-1]:
                save_name = target_examples[target_sample_idx[idx]].strip().split('/')[-1][:-4]
                if save_name in kl_set:
                    # print(idx)
                    count += 1
                    continue
                else:
                    save_all.append(save_name)
                # if len(save_all) == 60:
                #     break
            # cur_similar_name = list(target_examples[target_sample_idx[idx]].strip().split('/')[-1][:-4] for idx in cur_similarity[::-1])
            print(count)
            cur_similar_name =  list(dict.fromkeys(save_all))

            assert len(cur_similar_name) >= 50, "num of cur_similar_name is too small, please enlarge the similarity_idx size"

            # if i == 341:
            #     breakpoint()
            if img_name not in similarity_idx_dict:
                # print(i)
                similarity_idx_dict[img_name] = cur_similar_name[:50]

        # with open(f"{source_features_dir}/folder{foldid}_top50-similarity_add_400.json", "w") as outfile:
        #     json.dump(similarity_idx_dict, outfile)
        os.makedirs("./topall_json_add_for_re", exist_ok=True)
        with open("./topall_json_add_for_re/folder{}_top50-similarity_add_kl_{}.json".format(foldid,alpha), "w") as outfile:
            json.dump(similarity_idx_dict, outfile, indent=4)
        
