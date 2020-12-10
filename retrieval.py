import os
import csv
import sys
import shutil
import pprint
import argparse
import numpy as np

# 引数の取得
sys.path.append("./fast-reid")
from fastreid.engine import default_argument_parser
# 人物映像の検出
from detection_2 import HumanDetectionAndTracking
# 人物特徴の抽出
from person_feature import feature_extractor as person_feature_extractor
# 動作特徴の抽出
from action_feature import feature_extractor as action_feature_extractor


# 人物映像の抽出
def person_detection(args):
    output_dir = os.path.join(args.data_dir, "query_videos")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    detector = HumanDetectionAndTracking(
        input_video=args.query_video,
        output_dir=output_dir
    )
    query_videos = detector.detect_and_track_human()

    print("\n人物検出の結果")
    pprint.pprint(query_videos)


# コサイン類似度による検索
def cosine_similarity(query_fv, gallery_fvs):
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
    ranking = np.argsort(-similarity)[:args.person_ret_num]
    return ranking


# L2ノルムによる検索
def l2_similarity(query_fv, gallery_fvs):
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
    ranking = np.argsort(-similarity)[:args.action_ret_num]
    return ranking


# 検索のみを行う
def retrieval(args):
    # データの読み込み
    print("Reading retrieval data ...")
    action_fvs = np.load(args.gallery_action_features_npy)
    person_fvs = np.load(args.gallery_person_features_npy)
    with open(args.gallery_features_csv, "r") as f:
        data_list = [row for row in csv.reader(f)]
        
    # 人物の検索
    print("Retrieving with person features ...")
    person_query_fv = person_feature_extractor([args.query_video], args)[0]
    person_gallery_fvs = np.array([person_fvs[int(d[1])] for d in data_list])
    person_result = cosine_similarity(person_query_fv, person_gallery_fvs)
    refined_data_list = [data_list[i] for i in person_result]

    # 動作の検索
    print("Retrieving with action features ...")
    action_query_fv = action_feature_extractor([args.query_video])[0]
    action_gallery_fvs = np.array([action_fvs[int(d[0])] for d in refined_data_list])
    action_result = l2_similarity(action_query_fv, action_gallery_fvs)
    result_data_list = [refined_data_list[i] for i in action_result]
    
    # 検索結果の出力
    res_list = [d[2] for d in result_data_list]
    return res_list


if __name__ == "__main__":
    # args = parse_args()
    args = default_argument_parser().parse_args()
    args.gallery_features_csv = os.path.join(args.data_dir, "features.csv")
    args.gallery_action_features_npy = os.path.join(args.data_dir, "action_features.npy")
    args.gallery_person_features_npy = os.path.join(args.data_dir, "person_features.npy")
    res_list = retrieval(args)
    pprint.pprint(res_list)