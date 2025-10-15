
#!/bin/sh

# mkdir -p logs
# currenttime=`date "+%Y%m%d_%H%M%S"`

# feature_name=$1
# output_name=$2
# # BASE_DIR = /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wuwenxiao-240114050176/ranking_in_context-master/dataset/VOC2012

# # export CUDA_VISIBLE_DEVICES=1

# for folderid in 0 1 2 3 #2
# do
#     for seed in 0 #1 2 #3
#     do
#         export MASTER_PORT=$((12000 + $RANDOM % 20000))
#         # srun -p ntu --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -w SG-IDC2-10-51-5-46 \
#         python evaluate/evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir ./dataset \
#         --feature_name ${feature_name} --output_dir output_seg_images/${output_name}_folder${folderid}_seed${seed} \
#         --ckpt ./checkpoint-1000.pth --fold ${folderid} --seed ${seed} \
#         --annotaion_file 
#         # 2>&1 | tee -a logs/${folderid}-${seed}-${currenttime}.log
#         #  > /dev/null &
#         # echo -e "\033[32m[ Please check log: \"logs/${folderid}-${seed}-${currenttime}.log\" for details. ]\033[0m"
#     done
# done
# folderid=3
# seed=0
# python evaluate/evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wuwenxiao-240114050176/ranking_in_context-master/dataset --feature_name ${feature_name} --output_dir output_seg_images/${output_name}_folder${folderid}_seed${seed} --ckpt ./checkpoint-1000.pth --fold ${folderid} --seed ${seed} \
# 2>&1 | tee -a logs/${folderid}-${seed}-${currenttime}.log


mkdir -p logs
currenttime=`date "+%Y%m%d_%H%M%S"`

feature_name=$1
output_name=$2
# BASE_DIR = /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wuwenxiao-240114050176/ranking_in_context-master/dataset/VOC2012

# export CUDA_VISIBLE_DEVICES=1
seed=0
for folderid in 0 1 2 3 #2
do
    for trial in {1..10} #1 2 #3
    do
        export MASTER_PORT=$((12000 + $RANDOM % 20000))
        # srun -p ntu --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -w SG-IDC2-10-51-5-46 \
        python evaluate/evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir ./dataset \
        --feature_name ${feature_name} --output_dir output_seg_images/${output_name}_folder${folderid}_seed${seed} \
        --ckpt ./checkpoint-1000.pth --fold ${folderid} --seed ${seed} \
        --annotation_file ./ablation/only_rm_noleast/20250513_ori_filter_result_sourcefold${folderid}_fold${folderid}_rank5_10_dinov2_hodge_${trial}_hodge_repro_2e-4_8bs.json
        # 2>&1 | tee -a logs/${folderid}-${seed}-${currenttime}.log
        #  > /dev/null &
        # echo -e "\033[32m[ Please check log: \"logs/${folderid}-${seed}-${currenttime}.log\" for details. ]\033[0m"
    done
done