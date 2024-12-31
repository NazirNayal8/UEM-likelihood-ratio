import os
import argparse

from modeling.ood_segmentation import OoDSegmentationModel
from easydict import EasyDict as edict

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

    segmentor_ckpt = None
    if args.segmentor_ckpt is not None:
        segmentor_ckpt = args.segmentor_ckpt

    model = OoDSegmentationModel.load_from_checkpoint(
        ckpt, segmentor_ckpt=segmentor_ckpt)

    model.save_hyperparameters(overwrite_config(model.hparams, opts))

    datamodule = SemanticSegmentationDataModule(model.hparams)
    devices = 1
    if args.devices is not None:
        devices = [int(d) for d in args.devices.split(",")]
    output = Trainer(precision='32', devices=devices).test(
        model, datamodule=datamodule)

    return edict(output[0])


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

    if args.multiple_datasets:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]

    results = edict()
    ckpt_name = args.ckpt.split("/")[-1]
    results[ckpt_name] = edict()
    opts = ["DATA.DATASETS_FOLDER", args.datasets_folder]

    if args.segmentor_ckpt is not None:
        opts.extend(["MODEL.SEGMENTOR_CKPT", args.segmentor_ckpt])

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
        "--segmentor-ckpt",
        type=str,
        default=None,
        help="Path to the segmentation model checkpoint, if none then use the checkpoint stored in the model config"
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
