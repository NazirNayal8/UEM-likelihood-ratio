
import argparse
import torch
from modeling.segmentation import SegmentationModel
from easydict import EasyDict as edict
from datamodules import SemanticSegmentationDataModule
from pytorch_lightning import Trainer
from tools import overwrite_config

SEGMENTATION_DATASETS = ["cityscapes"]


def get_datamodule(args, hparams):

    if args.dataset == "cityscapes":

        datamodules = SemanticSegmentationDataModule(hparams)
        return datamodules

    else:
        raise ValueError(f"Undefined datamodule: {args.dataset}")


def main(args):

    # load config from ckpt
    ckpt = torch.load(args.ckpt)
    hparams = edict(ckpt["hyper_parameters"])
    state_dict = ckpt["state_dict"]
    model = SegmentationModel(hparams)
    model.load_state_dict(state_dict)

    hparams = overwrite_config(hparams, args.opts)

    # load datamodule

    datamodule = get_datamodule(args, hparams)

    devices = 1
    if args.devices is not None:
        devices = [int(d) for d in args.devices.split(",")]
    output = Trainer(devices=devices).test(model, datamodule=datamodule)

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation model")
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the segmentor model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Name of evaluation dataset"
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Devices to run the evaluation on"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line"
    )

    args = parser.parse_args()

    main(args)
