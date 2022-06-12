import os
import cv2
import torch
import joblib
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Optional, Union, List
from torchvision.datasets import VOCDetection
from torch.utils.data import SubsetRandomSampler

from .tokenizer import Tokenizer
from data.util import iou, resize_image, region_proposal
from .collate_fn import ft_collate_fn, ImgBatch


@dataclass
class ImgData:
    img: np.ndarray
    label: Optional[int] = None

    def __getitem__(self, idx: int):
        if idx:
            return self.label
        else:
            return self.img

    def __len__(self):
        return 2 - int(label is None)


@dataclass
class FeatureData:
    index: int
    feature: torch.Tensor
    label: Union[int, np.array]

    def __getitem__(self, idx: int):
        if idx and not idx % 2:
            return self.label
        elif not idx:
            return self.index
        else:
            return self.feature


class BaseDataset(Dataset):

    def __init__(self, cfg, tokenizer: Optional[Tokenizer] = None, file_ext='.jpg'):
        super().__init__()
        self.ext = file_ext
        self.img_size = cfg.img_size
        self.data_map = list()
        self.tokenizer = tokenizer
        self.path: Union[str, List[str]] = cfg.path

    def __getitem__(self, idx: int):
        return self._get(idx)

    def __len__(self):
        return len(self.data_map)

    def _get(self, idx):
        return self.data_map[idx]

    def _construct_dataset(self):
        if not isinstance(self.path, str):
            for path in self.path:
                self._construct_dataset_file(str(path))
        else:
            self._construct_dataset_file(self.path)

    def _construct_dataset_file(self, path):
        if not os.path.exists(path):
            raise ValueError('Bad Dataset File: %s' % path)

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                offset = f.tell() - len(line)
                handler = open(path, 'r', encoding='utf-8')
                handler.seek(offset)
                self.data_map.append(handler)
        f.close()

    def _parse_data(self, raw_data):
        raise NotImplementedError


class AlexnetDataset(BaseDataset):
    """
        Target data format: img_path img_label
        Note that the data label is of image-level
        And the data source is for image classification
    """

    def __init__(self, cfg, sep=' ', file_ext='.jpg'):
        super(AlexnetDataset, self).__init__(
            cfg, tokenizer=None, sep=sep, file_ext=file_ext)
        self._construct_dataset()

    def _parse_data(self, raw_data) -> ImgData:
        data = raw_data.strip().split(self.sep)
        assert len(data) <= 2
        img_path = os.path.join(self.data_root, data[0])
        try:
            label = int(data[1])
        except:
            label = None
        # robust processor for labeled / unlabeled dataset
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_img(img, self.img_size, self.img_size)
        img = np.asarray(img, dtype=np.float32)
        data = ImgData(img, label)
        return data


class FtDataset(BaseDataset):
    """
        target data format: img, List[object], 
        where object consists of regoin-level label and bounding box
        and the data source is for detection
        we treat negative sample as label=0, which means 'background' in dataset
        To construct pos-neg mixed data, we need to define a new __getitem__ method
        And we need another custom sampler to generate proper index
    """

    def __init__(self, cfg, tokenizer, file_ext='.jpg'):
        super(FtDataset, self).__init__(
            cfg, tokenizer=tokenizer, file_ext=file_ext)
        self.info = cfg.info
        self.threshold = cfg.threshold
        self.region_path = os.path.join(cfg.save_dir, 'ft.npy')
        self.pos_data_map, self.neg_data_map = list(), list()
        self.pos_iter, self.neg_iter = None, None
        if os.path.exists(self.region_path):
            self._load_ckpt()
        else:
            self._construct_dataset()
            self._merge()
            joblib.dump(self.data_map, self.region_path)
        self.pos_iter = iter(self.pos_sampler)
        self.neg_iter = iter(self.neg_sampler)

    def _construct_dataset(self):
        dataset = VOCDetection(root=self.path, **self.info)
        for idx in tqdm(range(len(dataset))):
            img, target = dataset[idx]
            img = np.array(img)  # RGB image
            objects = target['annotation']['object']
            if isinstance(objects, dict):
                objects = [objects]
            objects = self._parse_obj(objects)
            self._parse_data(img, objects)

    def _parse_obj(self, objects):
        res_objs = list()
        for obj in objects:
            label = self.tokenizer(obj['name'])
            if label < 0:
                raise ValueError
            bd_box = obj['bndbox']
            bd_box = [bd_box['xmin'], bd_box['ymin'],
                      bd_box['xmax'], bd_box['ymax']]
            bd_box = [int(x) for x in bd_box]
            res_objs.append({'label': label, 'bndbox': bd_box})
        return res_objs

    def _parse_data(self, img, objects):
        images, regions_pred, _ = region_proposal(img, self.img_size)
        for image, region_pred in zip(images, regions_pred):
            pos_flag = False
            for obj in objects:
                region_gold = obj['bndbox']
                iou_value = iou(region_gold, region_pred)
                if iou_value < self.threshold:
                    continue
                else:
                    self.pos_data_map.append(ImgData(image, obj['label']))
                    pos_flag = True
            if not pos_flag:
                self.neg_data_map.append(ImgData(image, 0))
        return None

    @property
    def pos_sampler(self):
        return SubsetRandomSampler(range(len(self.pos_data_map)))

    @property
    def neg_sampler(self):
        return SubsetRandomSampler(range(len(self.neg_data_map)))

    @property
    def pos_num(self):
        return len(self.pos_data_map)

    def __getitem__(self, idx: int):
        if idx < len(self.pos_data_amp):
            return self.pos_data_map[idx]
        else:
            idx -= len(self.pos_data_map)
            return self.neg_data_map[idx]

    def _merge(self):
        if len(self.data_map) < len(self.pos_data_map) + len(self.neg_data_map):
            self.data_map.clear()
            for data in self.pos_data_map:
                self.data_map.append(data)
            for data in self.neg_data_map:
                self.data_map.append(data)

    def _load_ckpt(self):
        self.data_map = joblib.load(self.region_path)
        for data in tqdm(self.data_map):
            if data.label == 0:
                self.neg_data_map.append(data)
            elif data.label > 0:
                self.pos_data_map.append(data)
            else:
                raise ValueError('Bad Data Label %s in Resuming.' % data.label)

    def get_batch(self, batch_size: int = 128, pos_prop: int = 4) -> ImgBatch:
        batch_idx = self._get_batch_idx(batch_size, pos_prop)
        batch = [self[id] for id in batch_idx]

        return ft_collate_fn(batch)

    def _get_batch_idx(self, batch_size: int = 128, pos_prop: int = 4):
        pos_num = int(batch_size / pos_prop)
        batch_idx = list()
        while len(batch_idx) < pos_num:
            try:
                pidx = next(self.pos_iter)
            except StopIteration:
                self.pos_iter = iter(self.pos_sampler)
                pidx = next(self.pos_iter)
            batch_idx.append(pidx)

        while len(batch_idx) < batch_size:
            try:
                nidx = next(self.neg_iter)
            except StopIteration:
                self.neg_iter = iter(self.neg_sampler)
                nidx = next(self.neg_iter) + self.pos_num
            batch_idx.append(nidx)
        return batch_idx


class SVMDataset(Dataset):
    """
        target data format: line[img_path rec_label rec_bbox]
        where label is of regoin-level
        and the data source is for SVM classification
        we also calculate the train data of bounding box regression in this part
    """

    def __init__(self, cfg, sep=' ', file_ext='.jpg'):
        super(SVMDataset, self).__init__(
            cfg, tokenizer=None, sep=sep, file_ext=file_ext)
        self.bbox_map = list()
        self.feature_map = defaultdict(list)
        self.label_map = defaultdict(list)
        self.svm_threshold = cfg.threshold
        self.bbox_threshold = cfg.reg_threshold
        self.svm_path = os.path.join(cfg.save_dir, 'svm.npy')
        self.bbox_path = os.path.join(cfg.save_dir, 'box.npy')
        self.model = None
        if os.path.exists(self.svm_path) and os.path.exists(self.bbox_path):
            self._load_ckpt()
            self._init_map()
        else:
            self._construct_dataset()
            joblib.save(self.data_map, self.svm_path)
            joblib.save(self.bbox_map, self.bbox_path)
            self._init_map()

    def _init_map(self):
        self.bbox_map.clear()
        self.feature_map.clear()
        self.label_map.clear()
        for data in tqdm(self.data_map):
            self.feature_map[data[0]].append(data[1])
            self.label_map[data[0]].append(data[2])

    def _load_ckpt(self):
        self.data_map = joblib.load(self.svm_path)

    def _parse_data(self, raw_data) -> None:
        data = raw_data.strip().split(self.sep)
        assert len(data) == 3

        img_path = os.path.join(self.data_root, data[0])
        try:
            label_gold = int(data[1])
        except:
            label_gold = None
        region_gold = data[2].split(',')
        region_gold = [int(x) for x in region_gold]

        images, regions_pred, _ = region_proposal(image_path, self.img_size)
        for image, region_pred in zip(images, regions_pred):
            iou_value = iou(region_gold, region_pred)
            if iou_value < self.svm_threshold:
                svm_label = 0
            else:
                svm_label = label_gold

            bbox_label = self._calc_bbox(region_pred, region_gold)
            image = torch.Tensor([image]).permute(0, 3, 1, 2)

            feature, _ = self.model(image)
            feature = feature.data.cpu().numpy()

            self.data_map.append(FeatureData(
                label_gold, feature[0], svm_label))
            self.bbox_map.append(FeatureData(label_gold, feature, bbox_label))

        return None

    def _calc_bbox(self, region_pred, region_gold, iou):
        box_label = np.zeros(5)
        px = float(region_pred[0]) + float(region_pred[4] / 2.0)
        py = float(region_pred[1]) + float(region_pred[5] / 2.0)
        ph = float(region_pred[5])
        pw = float(region_pred[4])

        gx = float(region_gold[0])
        gy = float(region_gold[1])
        gw = float(region_gold[2])
        gh = float(region_gold[3])

        box_label[1:5] = [(gx - px) / pw, (gy - py) / ph,
                          np.log(gw / pw), np.log(gh / ph)]
        if iou < self.bbox_threshold:
            box_label[0] = 0
        else:
            box_label[0] = 1
        return box_label

    def __getitem__(self, idx):
        return self.feature_map[idx], self.label_map[idx]


class RegDataset(Dataset):
    def __init__(self, cfg):
        super(RegDataset, self).__init__(cfg, tokenizer=None)
        self._load_ckpt(cfg.save_dir)

    def _load_ckpt(self, dir):
        path = os.path.join(dir, 'box.npy')
        assert os.path.exists(path)
        self.data_map = joblib.load(path)
