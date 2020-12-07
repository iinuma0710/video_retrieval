import sys
sys.path.append("./fast-reid")

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor, default_argument_parser, default_setup
from fastreid.utils.checkpoint import Checkpointer

import os
import cv2
import csv
import copy
import torch
import random
import numpy as np


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# 指定範囲から指定個数の整数をランダムに取得する
def get_randoms(minimum, maximum, num):
    random_list = []
    while len(random_list) < num:
        i = random.randint(minimum, maximum)
        if i not in random_list:
            random_list.append(i)
    return random_list


# 人物ごとの映像からランダムに num フレームを抜き出してくる
def get_inputs(video_path, num=1):
    # 映像を開く
    video = cv2.VideoCapture(video_path)

    # フレームに分解
    frame_list = []
    while True:
        ret, frame = video.read()
        if ret:
            frame_list.append(frame)
        else:
            video.release()
            break
    
    # ランダムにnumフレームを抜き出してくる
    # 最初の10フレームくらいは他の人が紛れてる子脳性があるので除外
    indexes = get_randoms(10, len(frame_list)-1, num)
    frames = np.array(frame_list)[indexes]
    # Fast-ReID の入力に合うように torch.tensor に変換して整形
    inputs = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2)
    return inputs


def read_features_csv(args):
    with open(args.csv, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        features_list = [row for row in reader]
    
    return features_list

# 人物特徴の抽出
def feature_extractor(video_path_list, args):
    # 初期設定
    # args = default_argument_parser().parse_args()
    args.config_file = "./fast-reid/configs/person_reid.yml"
    cfg = setup(args)

    # 特徴ベクトルの抽出
    fvs = []
    pred = DefaultPredictor(cfg)
    for video_path in video_path_list:
        input_frames = get_inputs(video_path)
        fv = pred(input_frames)
        fvs.append(np.mean(fv.numpy().copy(), axis=0))

    return fvs


if __name__ == "__main__":
    # 初期設定
    args = default_argument_parser().parse_args()
    args.config_file = "./fast-reid/configs/person_reid.yml"
    cfg = setup(args)
    features_list = read_features_csv(args)

    # ネットワークのロード
    pred = DefaultPredictor(cfg)

    # 繰り返し処理で特徴ベクトルを抽出
    person_fv_idx = 0
    new_infos = []
    person_fvs = []
    for info in features_list:
        # 人物の特徴ベクトルを抽出
        new_info = copy.deepcopy(info)
        inputs = get_inputs(info[5])
        fv = np.mean(pred(inputs).numpy().copy(), axis=0)
        person_fvs.append(fv)
        # 特徴ベクトル情報の更新
        new_info[4] = person_fv_idx
        person_fv_idx += 1
        new_infos.append(new_info)

        print(new_info)
    
    person_fvs = np.array(person_fvs)
    np.save(args.npy, person_fvs)
    with open(args.csv, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(new_infos)