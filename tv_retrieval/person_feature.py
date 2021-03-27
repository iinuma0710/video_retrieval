import sys
sys.path.append("../fast-reid")

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
    print(video_path)
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
    # 最初の10フレームくらいは他の人が紛れてる可能性があるので除外
    # print(len(frame_list)-1, video_path)
    indexes = get_randoms(10, len(frame_list)-1, num)
    frames = np.array(frame_list)[indexes]
    # Fast-ReID の入力に合うように torch.tensor に変換して整形
    inputs = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2)
    return inputs

def feature_extractor(tmp_dir, detected_dir, features_dir, args):
    # 処理結果の保存先
    features_csv = os.path.join(features_dir, "features.csv")
    features_npy = os.path.join(features_dir, "person_features.npy")

    # 初期設定
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    args.config_file = "../fast-reid/configs/person_reid.yml"
    cfg = setup(args)
    with open(features_csv, "r") as f:
        videos_list = [[r[0], r[1]] for r in csv.reader(f)]

    # ネットワークのロード
    pred = DefaultPredictor(cfg)
    
    # 繰り返し処理で特徴ベクトルを抽出
    person_fv_idx = 0
    new_video_list = []
    person_fvs = []
    for video_path, action_fv_idx in videos_list:
        # 人物の特徴ベクトルを抽出
        inputs = get_inputs(video_path)
        fv = np.mean(pred(inputs).numpy().copy(), axis=0)
        person_fvs.append(fv)
        # 特徴ベクトル情報の更新
        new_video_list.append([action_fv_idx, person_fv_idx, video_path])
        print(video_path, action_fv_idx, person_fv_idx)
        person_fv_idx += 1

    np.save(features_npy, np.array(person_fvs))
    with open(features_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerows(new_video_list)


def feature_extractor_from_video(video_path, args):
    # 初期設定
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    args.config_file = "../fast-reid/configs/person_reid.yml"
    cfg = setup(args)

    # 特徴ベクトルの抽出
    pred = DefaultPredictor(cfg)
    input_frames = get_inputs(video_path)
    fv = pred(input_frames)

    return np.mean(fv.numpy().copy(), axis=0)


if __name__ == "__main__":
    # 引数の整理
    args = default_argument_parser().parse_args()
    station = args.tv_station
    s_year = str(args.year)
    s_month = str(args.month).zfill(2)
    s_day = str(args.day).zfill(2)
    print("Processing {} programs of {}/{}/{}".format(station, s_day, s_month, s_year))

    # 必要なファイルの準備
    tmp_dir = "./data/tmp/"
    detected_dir = "./data/detected/{}/{}/{}_{}_{}_04_56/".format(station, s_year, s_year, s_month, s_day)
    features_dir = "./data/features/{}/{}/{}_{}_{}_04_56/".format(station, s_year, s_year, s_month, s_day)
    
    # 初期設定
    args = default_argument_parser().parse_args()
    
    feature_extractor(tmp_dir, detected_dir, features_dir, args)