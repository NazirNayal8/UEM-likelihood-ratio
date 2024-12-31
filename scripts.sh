#!/bin/bash


# Train a segmentor example
python train_segmentor.py --config configs/segmentation/cityscapes/discriminative.yaml \
SOLVER.BATCH_SIZE 4 \
DATA.DATASET_ROOT /BS/cityscapes00/Cityscapes

# Train a UEM example
python train_ood.py --config configs/segmentation/ood/g_g.yaml \
 MODEL.SEGMENTOR_CKPT ckpts/gmm-segmentor.ckpt \
 SOLVER.BATCH_SIZE 4 \
 DATA.DATASET_ROOT datasets/cityscapes \
 DATA.DATASETS_FOLDER datasets/ \
 DATA.COCO_ROOT datsets/coco

# Evaluate UEM example
 python evaluate_ood.py --ckpt "$ckpt" \
    --devices 0 \
    --datasets-folder datasets/ \
    --multiple-datasets \
    --out-path results/ \
    --dataset road_anomaly \
    --store-anomaly-scores \
    --segmentor-ckpt ckpts/d_d/linear-segmentor.ckpt \