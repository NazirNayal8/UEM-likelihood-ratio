# NCCL_P2P_DISABLE=1, python train_lr_gmm.py --config configs/segmentation/cityscapes/gmm_seg/mlp.yaml --name-suffix mlp_3_512_256_40ksched_16precision --distributed --multi-gpu-ids 0,1,2,3 \
# MODEL.BACKBONE.LEARNABLE_PARAMS.HIDDEN_DIM 512 \
# MODEL.BACKBONE.LEARNABLE_PARAMS.OUTPUT_DIM 256 \

# for testing
# NCCL_P2P_DISABLE=1, python train_lr_gmm.py --config configs/segmentation/cityscapes/gmm_seg/maskformer.yaml \
#  --name-suffix m2f --choose-gpu 0 --dev \


NCCL_P2P_DISABLE=1, python train_lr_gmm.py --config configs/segmentation/cityscapes/gmm_seg/maskformer.yaml \
 --name-suffix m2f_deepsup_1dl --distributed \
 --multi-gpu-ids 0,1,2,3 \
WANDB.ACTIVATE True \
MODEL.BACKBONE.FREEZE True \
SOLVER.BATCH_SIZE 4 \
SOLVER.NUM_WORKERS 6 \
SOLVER.MAX_STEPS 90000 \
MODEL.BACKBONE.LEARNABLE_PARAMS.TRANSFORMER_PREDICTOR_CONFIG.DEC_LAYERS 1 \


NCCL_P2P_DISABLE=1, python train_lr_gmm.py --config configs/segmentation/cityscapes/gmm_seg/maskformer.yaml \
 --name-suffix m2f_deepsup_lr_5e-5 --distributed \
 --multi-gpu-ids 0,1,2,3 \
WANDB.ACTIVATE True \
MODEL.BACKBONE.FREEZE True \
SOLVER.BATCH_SIZE 4 \
SOLVER.NUM_WORKERS 6 \
SOLVER.MAX_STEPS 90000 \
SOLVER.LR 5.0e-5 \



# MODEL.BACKBONE.LEARNABLE_PARAMS.TRANSFORMER_PREDICTOR_CONFIG.DEC_LAYERS 1 \


# for profiling:
# NCCL_P2P_DISABLE=1, python train_lr_gmm.py --config configs/segmentation/cityscapes/gmm_seg/conv_upscale_dinov2_b.yaml \
#  --name-suffix conv_upscale_2x_bs2_profiling --distributed \
#  --multi-gpu-ids 0,1,2,3 \
#  --profile-memory \
#  --profiling-output-file conv_upscale_dinov2_b_bs8_32p_val_afterpotentialfix \
# SOLVER.PRECISION 32 \
# WANDB.ACTIVATE False \
# SOLVER.MAX_STEPS 10 \
# SOLVER.BATCH_SIZE 8 \
# SOLVER.VAL_CHECK_INTERVAL 5



