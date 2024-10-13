import os
import argparse
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from modeling.ood_segmentation import OoDSegmentationModel
from easydict import EasyDict as edict
from datamodules.datasets.road_anomaly import RoadAnomaly
from datamodules.datasets.fishyscapes import FishyscapesLAF
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics import AUROC, AveragePrecision
from modeling.modules.metrics import FPR95
from tools import overwrite_config
from pytorch_lightning import Trainer
from datamodules import SemanticSegmentationDataModule


def evaluate_single_ckpt(ckpt, args, opts):
    """
    opts is a list of even size containing the hyperparameters to overwrite
    the model config. even value indices are the keys, and odd value indices
    are the values.
    """
    print(f"Evaluating {ckpt}")
    
    model = OoDSegmentationModel.load_from_checkpoint(ckpt)

    model.save_hyperparameters(overwrite_config(model.hparams, opts))
    
    datamodule = SemanticSegmentationDataModule(model.hparams)
    devices = 1
    if args.devices is not None:
        devices = [int(d) for d in args.devices.split(",")]
    output = Trainer(precision='32', devices=devices).test(model, datamodule=datamodule)

    return edict(output[0])

def evaluate_multiple_ckpts(args):

    opts = [
        "DATA.DATASETS_FOLDER", args.datasets_folder
    ]

    if args.multiple_datasets:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]

    ckpts = os.listdir(args.models_folder)
    results = edict()
    for ckpt in ckpts:
        results[ckpt] = edict()
        for dataset in datasets:
            results[ckpt][dataset] = evaluate_single_ckpt(
                ckpt=os.path.join(args.models_folder, ckpt), 
                args=args,
                opts=opts + ["DATA.EVAL_DATASET", dataset],
            )

    return results

def write_results(results, args):
    """
    Expected hierarchy of the results dictionary
    - model name
        - dataset name
            - metric name
    """

    if args.out_path is not None:
        # if args.out_path doesn't exist create it
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path, exist_ok=True)
        if args.multiple_models:
            out_path = os.path.join(args.models_folder, "all_results")
        else:
            model_folder = '/'.join(args.ckpt.split("/")[:-1])
            model_name = args.ckpt.split("/")[-1].split(".")[0]
            out_path = os.path.join(model_folder, model_name + "_results")

        # in the first row of the output, leave the first column empty for the checkpoint name
        # and print the metrics in the order given in the argument
        with open(f"{out_path}.txt", "w") as f:
            f.write("\t")
            dataset_list = list(results[list(results.keys())[0]].keys())
            for dataset in dataset_list:
                for metric in args.metrics_order.split(","):
                    f.write(f"{dataset}_{metric}\t")
            f.write("\n")
            for ckpt, ckpt_results in results.items():
                f.write(f"{ckpt}\t")
                for dataset, dataset_results in ckpt_results.items():
                    for metric in args.metrics_order.split(","):
                        f.write(f"{100 * dataset_results[metric]:.4f}\t")
                f.write("\n")

def main(args):

    
    
    if args.multiple_models:
        results = evaluate_multiple_ckpts(args)
    else:
        if args.multiple_datasets:
            datasets = args.dataset.split(",")
        else:
            datasets = [args.dataset]
        
        results = edict()
        ckpt_name = args.ckpt.split("/")[-1]
        results[ckpt_name] = edict()
        opts = ["DATA.DATASETS_FOLDER", args.datasets_folder]
        for dataset in datasets:
            results[ckpt_name][dataset] = evaluate_single_ckpt(
                args.ckpt,
                args, 
                opts + ["DATA.EVAL_DATASET", dataset]
            )

    write_results(results, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OoD Metrics")

    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="road_anomaly",
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--multiple-datasets",
        action="store_true",
        help="Evaluate multiple datasets, names given separated by commas in --dataset argument"
    )
    parser.add_argument(
        "--multiple-models", 
        action="store_true",
        help="Evaluate multiple checkpoints given in the --models-folder argument"
    )
    parser.add_argument(
        "--models-folder",
        type=str,
        default=None,
        help="Path to the folder containing multiple models"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Path to save the results as a text file"
    )
    parser.add_argument(
        "--metrics-order",
        type=str,
        default="llr_AUPR,llr_AUROC,llr_FPR95",
    )
    parser.add_argument(
        "--datasets-folder",
        type=str,
        default=None,
        help="Path to the evaluation datasets folder"
    )
    parser.add_argument(
        "--store-anomaly-scores",
        action="store_true",
        help="Store anomaly scores in the output"
    )

    args = parser.parse_args()

    main(args)