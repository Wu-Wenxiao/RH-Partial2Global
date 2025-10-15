# RH-Partial2Global
This repository contains the official implementation for the NeurIPS 2025 paper, "Towards Reliable and Holistic Visual In-Context Learning Prompt Selection".

---

### Step 1: Extract Features

First, extract deep features from both the **training (`trn`)** and **validation (`val`)** image sets using the pre-trained ViT model.

```bash
# Extract features for the training set
python ./tools/featextrater_folderwise_UnsupPR.py vit_large_patch14_clip_224.laion2b_ft_in12k features_vit-laion2b-in12k trn

# Extract features for the validation set
python ./tools/featextrater_folderwise_UnsupPR.py vit_large_patch14_clip_224.laion2b_ft_in12k features_vit-laion2b-in12k val
```

---

### Step 2: Calculate Initial Similarity

Next, compute the similarity scores. This is done for samples within the training set and between the validation and training sets.

```bash
# Calculate similarity within the training set
python tools/calculate_similarity.py features_vit-laion2b-in12k trn trn

# Calculate similarity between validation and training sets
python tools/calculate_similarity.py features_vit-laion2b-in12k val trn
```

---

### Step 3: Generate Training Set Annotations in Jackknife Manner

Run the evaluation script on the entire training set to generate the required annotations for the next steps.

```bash
# Run evaluation script
./evaluate/srun_seg_evaluate_all_iou.sh features_vit-laion2b-in12k_trn output_vit-laion2b_trn-all

# Consolidate annotations
python ./evaluate/get_annotation_all.py features_vit-laion2b-in12k_trn output_vit-laion2b_trn-all
```

---

### Step 4: Identify Samples for Removal Under the Guidance of Conformal Prediction

Use the KL divergence script to identify noisy or redundant samples from the training set that should be removed to improve prompt selection.

```bash
python ./tools/kl_cp_topk.py
```

---

### Step 5: Generate 

After identifying samples for removal, recalculate the similarity matrix for the refined training set.

```bash
python ./tools/calculate_similarity_rm_for_re.py features_vit-laion2b-in21k val trn
```

---

### Step 6: Generate Prompt Ranking

Navigate to the scripts directory and run the ranking script. You'll need to provide an `alpha` value and the `GPU ID`.

```bash
cd ./scripts_for_re

# Usage: ./run_alpha_ranking_single.sh <value_of_alpha> <GPU_ID>
# Example:
./run_alpha_ranking_single.sh 0.15 0
```

---

### Step 7: Evaluate Top-Ranked Sample

Finally, use the corresponding evaluation script to measure the performance using the top-ranked prompt sample generated in the previous step.

```bash
cd ./scripts_for_re

# Usage: ./run_alpha_eval_single.sh <value_of_alpha> <GPU_ID>
# Example:
./run_alpha_eval_single.sh 0.15 0
```