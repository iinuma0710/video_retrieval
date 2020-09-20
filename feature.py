import os
import sys
import csv
import torch
import numpy as np

from tools.test_net import test
import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.config.defaults import get_cfg
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter


def extract_features(
    cfg_file,
    shard_id=0,
    num_shards=1,
    init_method="tcp://localhost:9999",
    rng_seed=None,
    output_dir=None,
    opts=None
):
    """
    1. Config ファイルのセットアップ
    """
    # config ファイルを取得
    cfg = get_cfg()
    # 指定したファイルから設定を読み込む
    if cfg_file is not None:
        cfg.merge_from_file(cfg_file)
    # opts で渡される設定内容で上書きする
    if opts is not None:
        cfg.merge_from_list(opts)
    # Inherit parameters from args.
    if num_shards is not None and shard_id is not None:
        cfg.NUM_SHARDS = num_shards
        cfg.SHARD_ID = shard_id
    if rng_seed is not None:
        cfg.RNG_SEED = rng_seed
    if output_dir is not None:
        cfg.OUTPUT_DIR = output_dir
        # Create the checkpoint dir.
        cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
        
    """
    2. 映像特徴量の抽出
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                test,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=False,
        )
    else:
        test(cfg=cfg)


if __name__ == "__main__": 
    with open("/net/per610a/export/das18a/satoh-lab/share/datasets/eastenders/trecvid_2019/query/person_videos.csv", "r") as f:
        reader = csv.reader(f, delimiter=" ")
        video_list = [[row[2], row[0]] for row in reader]
    
    with open("tmp.csv", "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(video_list)

    logger = logging.get_logger(__name__)
    torch.multiprocessing.set_start_method("forkserver")
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    opts = [
        "TRAIN.ENABLE", False,
        "TEST.ENABLE", True,
        "DATA.PATH_TO_TEST_FILE", "tmp.csv",
        "DATA.SAMPLING_RATE", 8, # 32~64 frames -> 4, 65 frames ~ -> 8
        "TEST.NUM_ENSEMBLE_VIEWS", 4,
        "TEST.NUM_SPATIAL_CROPS", 3,
        "TEST.BATCH_SIZE", 12,
        "TEST.EXTRACT_FEATURES", True,
        "TEST.CHECKPOINT_FILE_PATH", "slowfast/checkpoints/SLOWFAST_8x8_R50_KINETICS600.pyth",
        "NUM_GPUS", 8,
        "FEATURES_FILE", "/net/per610a/export/das18a/satoh-lab/share/datasets/eastenders/trecvid_2019/query/feature.npy",
        "LABELS_FILE", "/net/per610a/export/das18a/satoh-lab/share/datasets/eastenders/trecvid_2019/query/label.npy"
    ]
    extract_features("slowfast/configs/Kinetics/SLOWFAST_8x8_R50.yaml", opts=opts)

    os.remove("tmp.csv")