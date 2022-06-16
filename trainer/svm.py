import os
import torch
import logging

from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader

from .base import Trainer


class SvmTrainer(Trainer):
    def __init__(self,
                 feature_model,
                 svm_model,
                 criterion,
                 optimizer,
                 config,
                 device,
                 data_module,
                 lr_scheduler=None):
        super(SvmTrainer, self).__init__(model=feature_model,
                                         criterion=criterion, optimizer=optimizer,
                                         config=config, device=device, data_loader=None,
                                         lr_scheduler=lr_scheduler, log_step=4)
        self.data_module = data_module
        self.svm = svm_model.to(device)
        self.idx = self.svm.class_num
        self._overall_indices = None
        self._current_train_indices = None
        self.pos_size, self.neg_size = 0, 0
        self.best_acc = (0, 1)
        if self.data_module.train_dataset is not None:
            self._init_data_loader()
        else:
            self.logger.error('No Training Data for SVM')
            raise ValueError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            for key, value in log.items():
                self.logger.debug('{:8s}: {}'.format(str(key), value))

            hard_indices = self.hard_negative_mining(epoch)
            self._update_data_loader(hard_indices)
        self._save_checkpoint(self.epochs + 1)

    def _train_epoch(self, epoch):
        self.model.eval()
        self.svm.train()
        log = dict()
        len_epoch = len(self.data_loader)
        for batch_idx, (batch, _) in enumerate(tqdm(self.data_loader)):
            data, label = batch.to(self.device)
            self.optimizer.zero_grad()
            feature, final = self.model(data)
            output = self.svm(feature)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            log.update({str(batch_idx): loss.item()})
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {}\t|\t Lr: {:.6f}\t|\tLoss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), self.optimizer.param_groups[0]['lr'], loss.item()))
            if batch_idx >= len_epoch:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def hard_negative_mining(self, epoch):
        self.model.eval()
        self.svm.eval()
        num_corrects = 0
        hard_negative_indexes = set()
        self.logger.info('Start Hard Negative Mining...')
        with torch.no_grad():
            for batch_idx, (batch, data_indexes) in enumerate(tqdm(self.val_data_loader)):
                data, labels = batch.to(self.device)
                feature, final = self.model(data)
                outputs = self.svm(feature)
                _, preds = torch.max(outputs, dim=1)
                num_corrects += torch.sum(preds == labels)
                fp_mask = (preds == 1)
                #tn_mask = (preds == 0)
                hard_negative_index = data_indexes[fp_mask]
                #easy_negative_index = data_indexes[tn_mask]
                hard_negative_indexes.update(set(hard_negative_index.tolist()))
        acc = num_corrects / len(self._overall_indices)
        if acc > self.best_acc[0]:
            self.best_acc = (acc, epoch)
            if epoch > 2:
                self._save_checkpoint(epoch)

        self.logger.info('Finished Validation.')
        self.logger.info('Validation Accuracy: %.5f, Found %d Hard Negative Samples' % (
            acc, len(hard_negative_indexes)))

        return hard_negative_indexes

    def _init_data_loader(self):
        self._overall_indices = set(
            range(len(self.data_module.train_dataset)))
        self.pos_size = self.data_module.train_dataset.pos_num
        self.neg_size = self.pos_size * self.config.neg_prop
        pos_idx, neg_idx = self.data_module.train_dataset.init_sampler_indices(
            self.neg_size)
        self._current_train_indices = set(pos_idx + neg_idx)
        self.logger.info(
            'Initial Train Dataset: %d Pos Samples: %d Neg Samples' % (self.pos_size, self.neg_size))
        sampler, _ = self.build_subset_sampler(
            self._current_train_indices, require_val=False)
        self.data_loader = self.data_module.train_dataloader(
            sampler=sampler)
        self.val_data_loader = self.data_module.val_dataloader()
        self.logger.info('DataLoader Building Done, Start Training...')

    def _update_data_loader(self, indices: set):
        self._current_train_indices |= indices
        self.neg_size = len(self._current_train_indices) - self.pos_size
        self.logger.info(
            'Current Train Dataset: %d Pos Samples: %d Neg Samples' % (self.pos_size, self.neg_size))
        sampler, _ = self.build_subset_sampler(
            self._current_train_indices, require_val=False)
        self.data_loader = self.data_module.train_dataloader(sampler=sampler)
        self.logger.info('DataLoader Building Done, Start Training...')

    def _save_checkpoint(self, epoch):
        model_dict = self.svm.state_dict()
        filename = os.path.join(
            self.save_dir, 'svm%d-epoch%d.pt' % (self.idx, epoch))
        torch.save(model_dict, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def get_remaining_indices(self, indices=None):
        if indices is None:
            indices = self._current_train_indices
        remain_indices = self._overall_indices - set(indices)
        return list(remain_indices)

    def build_subset_sampler(self, indices, require_val=False):
        train_sampler = SubsetRandomSampler(list(indices))
        if not require_val:
            return train_sampler, None
        val_indices = self.get_remaining_indices(indices)
        val_sampler = SubsetRandomSampler(val_indices)
        return train_sampler, val_sampler
