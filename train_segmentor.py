import os
import argparse
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

from easydict import EasyDict as edict
from functools import partial
from tools import (
    read_config,
    read_config_recursive,
    update_from_wandb,
    overwrite_config,
)
from pprint import pprint

from modeling.segmentation import SegmentationModel
from datamodules import DATAMODULES
from torchinfo import summary


def create_run_name(config, args):

    name = config.WANDB.RUN_NAME
    # name += f"_B:{config.MODEL.BACKBONE.NAME}"
    # name += f"_SH:{config.MODEL.SEGMENTATION_HEAD.NAME}"
    name += f"_{args.name_suffix}"

    return name


def main(args):

    torch.set_float32_matmul_precision("high")

    # read config
    config = edict(read_config_recursive(args.filename))

    # overwrite config with opts from args, each argument can be a string
    config = overwrite_config(config, args.opts)

    # make sure config parameters are consistent. Consistency rules are manually defined.
    config = SegmentationModel.apply_consistency(config, args)
    pprint(config)

    config.WANDB.RUN_NAME = create_run_name(config, args)

    # prepare logger
    if config.WANDB.ACTIVATE and not args.dev:
        logger = WandbLogger(
            name=config.WANDB.RUN_NAME, project=config.WANDB.PROJECT, config=config
        )
    else:
        logger = None
    if logger is not None:
        logger.experiment.name = config.WANDB.RUN_NAME

    # if running a sweep, update the config from the ones
    # given by the wandb agent
    if args.sweep_config is not None:
        config = update_from_wandb(config, edict(logger.experiment.config))

    limit_val_batches = 1.0
    if args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        # if memory is being profiled, run one validation batch only
        limit_val_batches = 1

    # for reproducibility
    seed_everything(config.RANDOM_SEED, workers=True)
    model = SegmentationModel(config)

    if args.summary:
        print(summary(model, input_size=(1, 3, 518, 1036), device="cuda"))

    datamodule = DATAMODULES[config.DATA.MODULE](config)

    ckpt_path = os.path.join(
        config.CKPT.DIR_PATH, config.DATA.NAME, config.WANDB.RUN_NAME
    )

    callbacks = [
        LearningRateMonitor() if not args.dev else ModelSummary(),
        ModelCheckpoint(
            save_top_k=1,
            dirpath=ckpt_path,
            monitor="val_iou",
            mode="max",
            filename="{epoch:02d}-{val_iou:.2f}",
            save_last=True,
        ),
    ]

    strategy = "auto"
    devices = 1

    if args.distributed:
        devices = -1
        if args.multi_gpu_ids is not None:
            devices = [int(i) for i in args.multi_gpu_ids.split(",")]
        strategy = "ddp"
    # allow for choosing gpu ID
    elif args.choose_gpu is not None:
        devices = [args.choose_gpu]

    # train
    trainer = Trainer(
        fast_dev_run=args.dev,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,  # ddp_find_unused_parameters_true
        log_every_n_steps=config.SOLVER.LOG_EVERY_N_STEPS,
        max_steps=config.SOLVER.MAX_STEPS,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=None,
        val_check_interval=config.SOLVER.VAL_CHECK_INTERVAL,
        precision=config.SOLVER.PRECISION,
        profiler="simple",
        limit_val_batches=limit_val_batches
    )

    resume_path = None
    if args.resume:
        resume_path = os.path.join(ckpt_path, "last.ckpt")
        if not os.path.exists(resume_path):
            resume_path = None

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)

    if args.profile_memory:
        try:
            torch.cuda.memory._dump_snapshot(
                f"snapshots/{args.profiling_output_file}.pickle"
            )
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to dump snapshot: {e}")

        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Trainer for GMM Likelihood Ratio")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="configs/gmm_likelihood_ratio/default.yaml",
    )

    parser.add_argument("--dev", action="store_true", help="Runs in Dev Mode")
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help="Sweep Config Path. If None don't run sweep, otherwise, run sweep.",
    )

    parser.add_argument("--choose-gpu", type=int,
                        default=None, help="Choose GPU")
    parser.add_argument(
        "--name-suffix", type=str, default="", help="Name suffix for the run"
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Run in distributed mode"
    )
    parser.add_argument(
        "--multi-gpu-ids",
        type=str,
        default=None,
        help="Multi GPU IDs for distributed mode",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print model summary",
    )

    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="run memory profiling",
    )
    parser.add_argument(
        "--profiling-output-file",
        type=str,
        default="memory_profile_gmmseg",
        help="Output file for memory profiling",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from last checkpoint",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()

    if args.sweep_config is None:
        main(args)
    else:
        import wandb

        sweep_config = read_config(args.sweep_config)
        sweep_id = wandb.sweep(sweep_config, project=sweep_config["project"])
        wandb.agent(sweep_id=sweep_id, function=partial(main, args))
