import torch
import hydra
import numpy as np

from config import args_util
from logger import setup_logging
from tester.base import Tester
from data.datamodule import DataModule
from model import model_factory, criterion_factory
from data.util import iou


def test(cfg):
    datamodule = DataModule(cfg=cfg.data, name=cfg.name)
    test_loader = datamodule.test_dataloader()
    cnn = model_factory['finetune'](cfg.model)
    svm_list = range(1, cnn.cfg.output_size)
    svms = [model_factory['svm'](cfg.model, x) for x in svm_list]
    criterion = criterion_factory[cfg.name]
    tester = Tester(feature_model=cnn, svm_models=svms,
                    criterion=criterion, config=cfg.trainer,
                    device=cfg.trainer.device, data_loader=test_loader)
    tester.test()


@hydra.main(config_path='../config', config_name='test')
def main(configs):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path,
                  file_name='test.log')
    torch.set_printoptions(precision=5)
    test(cfg=configs)


if __name__ == '__main__':
    args_util()
    main()
