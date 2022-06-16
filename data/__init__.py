from .dataset import AlexnetDataset, FtDataset, SVMDataset, RegDataset
from .collate_fn import alex_collate_fn, ft_collate_fn, reg_collate_fn


dataset_factory = {
    'alex': AlexnetDataset,
    'finetune': FtDataset,
    'svm': SVMDataset,
    'reg': RegDataset,
}

collate_factory = {
    'alex': alex_collate_fn,
    'finetune': ft_collate_fn,
    'reg': reg_collate_fn,
    'svm': svm_collate_cn,
}
