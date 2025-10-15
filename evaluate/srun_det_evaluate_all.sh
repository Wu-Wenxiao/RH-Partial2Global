
#!/bin/sh

mkdir -p logs
currenttime=`date "+%Y%m%d_%H%M%S"`

feature_name=$1
output_name=$2


for folderid in 0 #2
do
    for seed in 0 #1 2 #3
    do
        for trial in 1
        do
            CUDA_VISIBLE_DEVICES=3 python evaluate/evaluate_segmentation_det_all.py --task detection --dataset_type pascal_det --model mae_vit_large_patch16 --base_dir ./dataset \
            --feature_name ${feature_name} --output_dir output_seg_images_ranking/${output_name}_folder${folderid}_seed${seed} \
            --ckpt ./checkpoint-1000.pth --seed ${seed} \
            --annotation_file ./annotation_xu/det_feature/train-all-similarity-dict.json
        done
    done
done