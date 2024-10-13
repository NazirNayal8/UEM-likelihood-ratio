
from easydict import EasyDict as edict
from .features import SAMFeaturesDataModule
from .mini_tasks import MiniTaskDataModule
from .semantic_segmentation import SemanticSegmentationDataModule


DATAMODULES = edict(
    SAM_FEATURES=SAMFeaturesDataModule,
    MINI_TASK=MiniTaskDataModule,
    SemanticSegmentation=SemanticSegmentationDataModule
)