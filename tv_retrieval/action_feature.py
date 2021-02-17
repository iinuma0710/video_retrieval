import os
import sys
import csv
import glob
import math
import torch
import argparse
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

TV_STATIONS = ["bs1", "etv", "fuji", "net", "NHK", "ntv", "tbs", "tvtokyo"]
PROCESSING_UNIT_NUM = 1000


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


# 特徴抽出に使う一時ファイルの作成
def prepare_tmp_files(detected_dir, tmp_dir):
    detected_videos = glob.glob(os.path.join(detected_dir, "*.mp4"))
    unit_num = math.ceil(len(detected_videos) / PROCESSING_UNIT_NUM)

    res_list = []
    for i in range(unit_num):
        s = i * PROCESSING_UNIT_NUM
        t = min((i + 1) * PROCESSING_UNIT_NUM, len(detected_videos))
        tmp_videos = [[detected_videos[idx], idx] for idx in range(s, t)]
        tmp_csv = os.path.join(tmp_dir, "tmp_{}.csv".format(i))
        tmp_features = os.path.join(tmp_dir, "features_{}.npy".format(i))
        tmp_labels = os.path.join(tmp_dir, "labels_{}.npy".format(i))
        
        res_list.append([tmp_csv, tmp_features, tmp_labels])
        with open(os.path.join(tmp_dir, "tmp_{}.csv".format(i)), "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(tmp_videos)

    return res_list


# 結果をまとめる
def merge_result(features_dir, tmp_files):
    action_features = []
    features_csv = []
    action_fv_idx = 0
    
    for tmp_csv, features_npy, labels_npy in tmp_files:
        print("Merging ", tmp_csv)
        features = np.load(features_npy)
        labels = np.load(labels_npy)
        with open(tmp_csv, "r") as f:
            videos = [r for r in csv.reader(f, delimiter=" ")]

        for path, idx in videos:
            fv = np.mean(features[np.where(labels[:, 1] == int(idx))], axis=0)
            norm_fv = fv / np.sqrt(np.sum(fv ** 2))
            action_features.append(norm_fv)
            features_csv.append([path, action_fv_idx])
            action_fv_idx += 1
    
    np.save(os.path.join(features_dir, "action_features.npy"), np.array(action_features))
    with open(os.path.join(features_dir, "features.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(features_csv)


def feature_extractor(tmp_dir, detected_dir, features_dir, num_gpu=6, visible_devices="2,3,4,5,6,7"):
    tmp_files = prepare_tmp_files(detected_dir, tmp_dir)
    
    # 特徴抽出
    logger = logging.get_logger(__name__)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    for tmp_csv, features_npy, labels_npy in tmp_files:
        print("Processing : ", tmp_csv)
        opts = [
            "TRAIN.ENABLE", False,
            "TEST.ENABLE", True,
            "DATA.PATH_TO_TEST_FILE", tmp_csv,
            "DATA.SAMPLING_RATE", 8, # 32~64 frames -> 4, 65 frames ~ -> 8
            "TEST.NUM_ENSEMBLE_VIEWS", 4,
            "TEST.NUM_SPATIAL_CROPS", 3,
            "TEST.BATCH_SIZE", 12,
            "TEST.EXTRACT_FEATURES", True,
            "TEST.CHECKPOINT_FILE_PATH", "../slowfast/checkpoints/SLOWFAST_8x8_R50_KINETICS600.pyth",
            "NUM_GPUS", num_gpu,
            "FEATURES_FILE", features_npy,
            "LABELS_FILE", labels_npy
        ]
        extract_features("../slowfast/configs/Kinetics/SLOWFAST_8x8_R50.yaml", opts=opts)
    
    # 一つのファイルにまとめる
    merge_result(features_dir, tmp_files)


def feature_extractor_from_video(video_path):
    with open("./data/tmp/query_tmp.csv", "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows([[video_path, 0]])

    # 特徴抽出
    logger = logging.get_logger(__name__)
    # torch.multiprocessing.set_start_method("forkserver")
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    opts = [
        "TRAIN.ENABLE", False,
        "TEST.ENABLE", True,
        "DATA.PATH_TO_TEST_FILE", "./data/tmp/query_tmp.csv",
        "DATA.SAMPLING_RATE", 8,
        "TEST.NUM_ENSEMBLE_VIEWS", 4,
        "TEST.NUM_SPATIAL_CROPS", 3,
        "TEST.BATCH_SIZE", 12,
        "TEST.EXTRACT_FEATURES", True,
        "TEST.CHECKPOINT_FILE_PATH", "../slowfast/checkpoints/SLOWFAST_8x8_R50_KINETICS600.pyth",
        "NUM_GPUS", 1,
        "FEATURES_FILE", "./data/tmp/query_feature.npy",
        "LABELS_FILE", "./data/tmp/query_label.npy"
    ]
    extract_features("../slowfast/configs/Kinetics/SLOWFAST_8x8_R50.yaml", opts=opts)

    # 後始末
    fvs = np.load("./data/tmp/query_feature.npy")
    norm_fv = np.mean(fvs, axis=0)
    os.remove("./data/tmp/query_tmp.csv")
    os.remove("./data/tmp/query_feature.npy")
    os.remove("./data/tmp/query_label.npy")

    return norm_fv


# 引数の整理
def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--tv_station',
                        type=str,
                        choices=TV_STATIONS,
                        required=True,
                        help="TV Station"
                       )
    parser.add_argument('-y', '--year',
                        type=int,
                        default=2021,
                        help="Year"
                       )
    parser.add_argument('-m', '--month',
                        type=int,
                        default=2,
                        help="Year"
                       )
    parser.add_argument('-d', '--day',
                        type=int,
                        default=7,
                        help="Year"
                       )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 引数の整理
    args = parse_arg()
    station = args.tv_station
    s_year = str(args.year)
    s_month = str(args.month).zfill(2)
    s_day = str(args.day).zfill(2)
    print("Processing {} programs of {}/{}/{}".format(station, s_day, s_month, s_year))

    # 実行環境の準備
    torch.multiprocessing.set_start_method("forkserver")

    # 必要なファイルの準備
    tmp_dir = "./data/tmp/"
    detected_dir = "./data/detected/{}/{}/{}_{}_{}_04_56/".format(station, s_year, s_year, s_month, s_day)
    features_dir = "./data/features/{}/{}/{}_{}_{}_04_56/".format(station, s_year, s_year, s_month, s_day)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    
    feature_extractor(tmp_dir, detected_dir, features_dir)
