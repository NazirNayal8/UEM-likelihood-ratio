

# for fast testing:

# NCCL_P2P_DISABLE=1, python finetune_ood.py --config configs/segmentation/ood/augmented_linear.yaml \
#  --choose-gpu 0 --dev \
#  --name-suffix ood_prob1.0_augmented_linear \
#  DATA.OOD_PROB 1.0 \
#  SOLVER.LR_SCHEDULER.NUM_WARMUP_STEPS 0 \
#  MODEL.OOD_HEAD.UPDATE_INTERVAL 3 \
#  SOLVER.MAX_STEPS 4000 \
#  MODEL.OOD_HEAD.NUM_COMPONENTS_PER_CLASS 5 \
#  DATA.COCO_PROXY_SIZE 30000 \
#  SOLVER.VAL_CHECK_INTERVAL 50 \
#  SOLVER.LOG_EVERY_N_STEPS 25 \
#  SOLVER.NUM_WORKERS 4 \

NCCL_P2P_DISABLE=1, python finetune_ood.py --config configs/segmentation/ood/augmented_linear.yaml \
 --distributed --multi-gpu-ids 0,1,2,3 \
 --name-suffix D-D_m2f_last_4kiter_gmmlossw1.0 \
 WANDB.RUN_NAME auGmented \
 MODEL.SEGMENTOR_CKPT ckpts/segmentation/cityscapes/GMMSeg_m2f_deepsup/last.ckpt \
 SOLVER.BATCH_SIZE 4 \
 SOLVER.VAL_CHECK_INTERVAL 50 \
 MODEL.LOSS.GMM_LOSS_WEIGHT 1.0 \
#  SOLVER.MAX_STEPS 12000 \
#  --sweep-config configs/segmentation/ood/augmented_sweep.yaml \


# for gmm inlier experiments
# NCCL_P2P_DISABLE=1, python finetune_ood.py --config configs/segmentation/ood/augmented_linear.yaml \
#  --distributed --multi-gpu-ids 0,1,2,3,4,5,6,7 \
#  --name-suffix ood_prob1.0_augmented_linear_LOSS_1_LLR_3_comp5_warmup1500steps_gmmloss_0.0 \
#  DATA.OOD_PROB 1.0 \
#  SOLVER.MAX_STEPS 4000 \
#  DATA.COCO_PROXY_SIZE 30000 \
#  SOLVER.VAL_CHECK_INTERVAL 50 \
#  SOLVER.LOG_EVERY_N_STEPS 25 \
#  SOLVER.NUM_WORKERS 4 \
#  DATA.EVAL_DATASET road_anomaly \
#  MODEL.LOSS.GMM_LOSS_WEIGHT 0.0 \
#  SOLVER.LR_SCHEDULER.NUM_WARMUP_STEPS 500 \


#  MODEL.OOD_HEAD.UPDATE_INTERVAL 3 \
#  MODEL.OOD_HEAD.NUM_COMPONENTS_PER_CLASS 5 \

#  MODEL.OOD_HEAD.UPDATE_INTERVAL 50 \
#  MODEL.OOD_HEAD.GAMMA_MEAN 0.8 \
#  MODEL.OOD_HEAD.MEMORY_SIZE 200 \

#  SOLVER.LR 1.0e-5 \