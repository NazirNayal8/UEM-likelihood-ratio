import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .datasets.lc_sam import LCSAM
from easydict import EasyDict as edict

class SAMFeaturesDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        if self.config.DATA.NAME == "lc_sam":

            train_config = edict(
                DATASET_ROOT=self.config.DATA.ROOT,
                EXCLUDED_FOLDERS=self.config.DATA.EXCLUDED_FOLDERS,
                EXCLUDED_LABELS=self.config.DATA.EXCLUDED_LABELS,
                MODE="train",
                SHAPE_MODE=self.config.DATA.SHAPE_MODE, 
                SAM_MODE=self.config.DATA.SAM_MODE,
            )
            valid_config = edict(
                DATASET_ROOT=self.config.DATA.ROOT,
                EXCLUDED_FOLDERS=self.config.DATA.EXCLUDED_FOLDERS,
                EXCLUDED_LABELS=self.config.DATA.EXCLUDED_LABELS,
                MODE="val",
                SHAPE_MODE=self.config.DATA.SHAPE_MODE,
                SAM_MODE=self.config.DATA.SAM_MODE,
            )

            self.train_dataset = LCSAM(train_config)
            self.valid_dataset = LCSAM(valid_config)

        else:
            raise ValueError(f"Undefined Dataset: {self.config.DATA.NAME}")


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.DATA.BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.config.DATA.NUM_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.config.DATA.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.DATA.NUM_WORKERS
        )

    def test_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.config.DATA.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.DATA.NUM_WORKERS
        )