cd ..
set -x
set -e

# LOG_FILE="./log_for_re/test_log/test_alpha_rh_part.txt"
# exec &> >(tee -a "$LOG_FILE")

# for alpha in {0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8}
for alpha in "$@"
do
# alpha=$1
# CUDA_ID=$2
echo "" # 添加空行以提高可读性
echo "############################################################"
echo "##  开始新的 alpha 循环, 当前 alpha = ${alpha}"
echo "############################################################"
for source_fold_idx in {0,1,2,3}
do
    echo ""
    echo "========================================"
    echo "    [alpha=${alpha}] 开始处理 source_fold_idx = ${source_fold_idx}"
    echo "========================================"
    for i in {1..10}
    do
        seed=0
        feature_name=features_vit-laion2b-in21k_val
        output_name=./log_for_re/output_log/fold${source_fold_idx}_alpha${alpha}_trial${i}_seed${seed}_${feature_name}
        ANNOTATION_FILE=./output_json_for_re_rh/sourcefold${source_fold_idx}_fold${source_fold_idx}_alpha${alpha}_rank5_10_dinov2_hodge_${i}_rh.json
        echo "    -> [alpha=${alpha}, fold=${source_fold_idx}] 开始第 ${i} 次测试..."
        echo "    -> 读取排序结果文件${ANNOTATION_FILE}..."
        # export CUDA_ID=0
        CUDA_VISIBLE_DEVICES=${CUDA_ID} python evaluate/evaluate_segmentation_for_re.py --model mae_vit_large_patch16 --base_dir ./dataset \
        --feature_name ${feature_name} --output_dir ${output_name} --fold ${source_fold_idx} \
        --ckpt ./checkpoint-1000.pth --seed ${seed} --annotation_file ${ANNOTATION_FILE} 

        echo "    -> [alpha=${alpha}, fold=${source_fold_idx}] 第 ${i} 次测试完成。"
        echo "    ------------------------------------"
    done

done

done