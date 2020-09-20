import os
import csv
import random
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.resister()
class KineticsTriplet(torch.utils.data.Dataset):
    """
    datasets/kinetics.py をベースに，Triplet Margin Loss で使えるデータローダを作成する
    """

    def __init__(self, cfg, mode, num_retries=10):
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        if self.mode == "train" or self.mode == "val":
            # train, val のときは動画からランダムに1クリップのみ抽出する
            self._num_clips = 1
        elif self.mode == "test":
            # test モードのときは同じ動画から複数のクリップを抽出する
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        else:
            # mode は train, val, test のいずれか
            assert mode in ["train", "val", "test"], "Split '{}' not supported for Kinetics".format(mode)

        logger.info("Constructing Kinetics Triplet Loader {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        # 各 mode で path_to_file を指定する
        if self.mode == "train":
            path_to_file = self.cfg.DATA.PATH_TO_TRAIN_FILE
        elif self.mode == "val":
            path_to_file = self.cfg.DATA.PATH_TO_VAL_FILE
        elif self.mode == "test":
            path_to_file = self.cfg.DATA.PATH_TO_TEST_FILE
        # ファイルの存在確認
        assert PathManager.exists(path_to_file), "{} not found".format(path_to_file)

        # データファイルの読み込み
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as fp:
            reader = csv.reader(fp, delimiter=" ")
            for clip_idx, path_label in enumerate(reader):
                path, label = path_label
                for idx in range(self._num_clips):
                    self._path_to_videos.append(path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = []
        
        # 読み込みに失敗した場合 (self._path_to_videos が空の場合)
        assert len(self._path_to_videos) > 0, "Failesd to load data from {}".format(path_to_file)
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )
    
    def __getitem__(self, index):
        """
        index から映像を読み込んでフレーム画像とラベル，映像のインデックスを返す
        """
        # short cycle が使われている場合は, input はタプルで渡される．
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode == "train" or self.mode == "val":
            # train, val のときは random cropping
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S)
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )
        elif self.mode == "test":
            pass