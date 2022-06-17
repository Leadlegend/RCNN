from .dataset import *
from .collate_fn import *


dataset_factory = {
    'alex': AlexnetDataset,
    'finetune': FtDataset,
    'svm': SVMDataset,
    'reg': RegDataset,
    'test': TestDataset,
}

collate_factory = {
    'alex': alex_collate_fn,
    'finetune': ft_collate_fn,
    'reg': reg_collate_fn,
    'svm': svm_collate_fn,
    'test': None,
}
