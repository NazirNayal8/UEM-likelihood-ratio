# NCCL_P2P_DISABLE=1, python eval_segmentor.py --ckpt ckpts/segmentation/cityscapes/GMM_seg_dinov2_b_fpn_1_4_8_11_bs2-bf16-mixed/epoch=258-val_iou=0.81.ckpt --dataset cityscapes \
#  --devices 0,1,2,3,4,5,6,7

# CKPTS=(
#     "ckpts/ood_segmentation/cityscapes_ood/gmmseg_ood_head_ood_prob1.0_augmented_LOSS_1_LLR_3_comp5/step=3200-llr_AUPR=0.6671.ckpt"
#     "ckpts/ood_segmentation/cityscapes_ood/gmmseg_ood_head_ood_prob1.0_augmented_LOSS_1_LLR_3_comp5/step=3200-llr_FPR95=0.1013.ckpt"
#     "ckpts/ood_segmentation/cityscapes_ood/gmmseg_ood_head_ood_prob1.0_augmented_LOSS_1_LLR_3_comp5/last.ckpt"
# )

# #ckpt="ckpts/ood_segmentation/cityscapes_ood/gmmseg_ood_head_ood_prob1.0_augmented_LOSS_2_LLR_3_comp5/last.ckpt"

# for ckpt in "${CKPTS[@]}"; do 
#     echo "Evalauting $ckpt Road Anomaly"
#     NCCL_P2P_DISABLE=1, python evaluate_ood.py --ckpt "$ckpt" \
#     --devices 0,1,2,3 \
#     --datasets-folder /scratch/users/nnayal21/hpc_run/datasets/ \
#     DATA.EVAL_DATASET road_anomaly \
#     DATA.DATASETS_FOLDER /scratch/users/nnayal21/hpc_run/datasets/
#     # python evaluate_ood.py --ckpt ckpts/ood_segmentation/cityscapes_ood/gmmseg_ood_head_/last.ckpt
#     echo "Evalauting $ckpt fishyscapes_laf"
#     NCCL_P2P_DISABLE=1, python evaluate_ood.py --ckpt "$ckpt" \
#     --devices 0,1,2,3 \
#     --datasets-folder /scratch/users/nnayal21/hpc_run/datasets/ \
#     DATA.EVAL_DATASET fishyscapes_laf \
#     DATA.DATASETS_FOLDER /scratch/users/nnayal21/hpc_run/datasets/
# done 


ckpt="ckpts/ood_segmentation/cityscapes_ood/gmmseg_ood_head_ood_prob1.0_augmented_linear_LOSS_1_LLR_3_comp5_warmup1500steps_gmmloss_0.0/step=1450-llr_AUPR=0.9331.ckpt"

NCCL_P2P_DISABLE=1, python evaluate_ood.py --ckpt "$ckpt" \
    --devices 0,1,2,3 \
    --datasets-folder /scratch/users/nnayal21/hpc_run/datasets/ \
    --multiple-datasets \
    --out-path results/ood_map/ \
    --dataset road_anomaly,fishyscapes_laf \
    --multiple-models \
    --models-folder ckpts/ood_segmentation/cityscapes_ood/auGmented_D-D_m2f_last_4kiter_gmmlossw0.5_nc5_ui3_memsz8000_projnl3_projhd512 \


NCCL_P2P_DISABLE=1, python evaluate_ood.py --ckpt "$ckpt" \
    --devices 0,1,2,3 \
    --datasets-folder /scratch/users/nnayal21/hpc_run/datasets/ \
    --multiple-datasets \
    --out-path results/ood_map/ \
    --dataset road_anomaly,fishyscapes_laf \
    --multiple-models \
    --models-folder ckpts/ood_segmentation/cityscapes_ood/auGmented_D-D_m2f_last_4bs_4kiter_nc5_ui3_memsz8000_projnl3_projhd512 \


# NCCL_P2P_DISABLE=1, python evaluate_ood.py --ckpt "$ckpt" \
#     --devices 0,1,2,3 \
#     --datasets-folder /scratch/users/nnayal21/hpc_run/datasets/ \
#     --multiple-models \
#     --multiple-datasets \
#     --models-folder ckpts/ood_segmentation/cityscapes_ood/gmmseg_ood_head_ood_prob1.0_augmented_linear-inlier-vs-linear-OODHead_v1_gmmloss0.0_orig_warmup \
#     --out-path results/ood/ \
#     --dataset road_anomaly