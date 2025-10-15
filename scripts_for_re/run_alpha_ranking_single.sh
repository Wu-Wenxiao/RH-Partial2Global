#!/bin/bash
alpha=$1
ID=$2
set -x
set -e
cd ..
LOG_FILE="./log_for_re/single_log/run_alpha_${alpha}_noadd.txt"

# 使用 tee 命令将所有输出同时打印到屏幕和日志文件
# 将整个脚本的输出重定向
exec &> >(tee -a "$LOG_FILE")
# for alpha in {0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45}

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
        echo "    -> [alpha=${alpha}, fold=${source_fold_idx}] 开始第 ${i} 次测试..."
            export CUDA_ID=${ID}
            python test_hodge_ori_remove_cover_v1_for_re.py --source_fold_idx $source_fold_idx \
            --fold_idx $source_fold_idx --n_gallery_list 5 10 \
            --backbone dinov2 --suffix ${i}_rh --ckpt_root ./ckpt_xu/all \
            --data_root ./dataset/VOC2012/JPEGImages/ \
            --alpha ${alpha} \
            --similarity_annotation ./dataset/VOC2012/features_vit-laion2b-in21k_val/folder${source_fold_idx}_top50-similarity.json
            # --similarity_annotation ./topall_json_add_for_re/folder${source_fold_idx}_top50-similarity_add_kl_${alpha}.json

        echo "    -> [alpha=${alpha}, fold=${source_fold_idx}] 第 ${i} 次测试完成。"
        echo "    ------------------------------------"
    done

done

