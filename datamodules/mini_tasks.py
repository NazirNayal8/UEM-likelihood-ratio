import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from .datasets.sorting import SortDataset 
from torch.utils.data import DataLoader

class MiniTaskDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        if self.config.DATA.NAME == "sorting":

            assert self.config.MODEL.VOCAB_SIZE == self.config.DATA.NUM_DIGITS, (
                f"Vocab size ({self.config.MODEL.VOCAB_SIZE}) must be equal to "
                f"number of digits ({self.config.DATA.NUM_DIGITS})"
            )

            assert self.config.MODEL.BLOCK_SIZE == 2 * self.config.DATA.LENGTH, (
                f"Block size ({self.config.MODEL.BLOCK_SIZE}) must be equal to "
                f"2 * sequence length ({2 * self.config.DATA.LENGTH})"
            )

            self.train_dataset = SortDataset(
                split='train',
                length=self.config.DATA.LENGTH,
                num_digits=self.config.DATA.NUM_DIGITS,
            )

            self.valid_dataset = SortDataset(
                split='test',
                length=self.config.DATA.LENGTH,
                num_digits=self.config.DATA.NUM_DIGITS,
            )
        else:
            raise ValueError(f"Undefined Dataset: {self.config.DATA.NAME}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
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