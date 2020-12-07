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


# class NoFileExistsError(Exception):
#     """ 必要なファイルがなかったときに呼び出されるエラー """

# # コマンドライン引数から検索の設定を取得
# def parse_args():
#     # コマンドライン引数
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str)
#     parser.add_argument('--query_video', type=str)
#     parser.add_argument('--person_ret_num', type=int, default=5000)
#     parser.add_argument('--action_ret_num', type=int, default=100)
#     parser.add_argument('--add_to_gallery', action="store_true")
#     parser.add_argument('--save_result', action="store_true")
#     parser.add_argument('--person_detection', action="store_true")
#     args = parser.parse_args()

#     # 検索に必要なファイルのパスを追加
#     args.gallery_features_csv = os.path.join(args.data_dir, "features.csv")
#     args.gallery_action_features_npy = os.path.join(args.data_dir, "action_features.npy")
#     args.gallery_person_features_npy = os.path.join(args.data_dir, "person_features.npy")

#     # ファイルの存在確認
#     if not os.path.exists(args.gallery_features_csv):
#         raise NoFileExistsError("{} does not exist.".format(args.gallery_features_csv))
#     if not os.path.exists(args.gallery_person_features_npy):
#         raise NoFileExistsError("{} does not exist.".format(args.gallery_person_features_npy))
#     if not os.path.exists(args.gallery_action_features_npy):
#         raise NoFileExistsError("{} does not exist.".format(args.gallery_action_features_npy))

#     return args

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


# 人物の検索
def person_retrieval(args):
    # 特徴ベクトルの取得
    print("Extracting a person feature vector with FastReID")
    query_fv = person_feature_extractor(args.query_video)
    tmp_fvs = np.load(args.gallery_person_features_npy)
    with open(args.gallery_features_csv, "r") as f:
        gallery_fvs = np.array([tmp_fvs[int(row[1])] for row in csv.reader(f)])
    
    # コサイン類似度で検索
    print("Retrieving person with cosine similarity")
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
    ranking = np.argsort(-similarity)[:args.person_ret_num]

    return ranking

# 動作の検索
def action_retrieval(args, person_result):
    # 特徴ベクトルの取得
    print("Extracting an action feature vector with SlowFast Networks")
    query_fv = action_feature_extractor(args.query_video)
    tmp_fvs = np.load(args.gallery_action_features_npy)
    with open(args.gallery_features_csv, "r") as f:
        rows = [row for idx, row in enumerate(csv.reader(f)) if idx in person_result]
    action_videos = [r[2] for r in rows]
    gallery_fvs = np.array([tmp_fvs[int(r[0])] for r in rows])
    
    # L2類似度で検索
    print("Retrieving person with L2 similarity")
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
    ranking = np.argsort(-similarity)[:args.action_ret_num]

    # 検索結果の映像へのパスを取得
    result_video_path_list = [action_videos[i] for i in ranking]
    return ranking, result_video_path_list


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
    pprint.pprint(res_list)


# 検出と検索を行う
def retrieval_with_detection(args):
    # 映像の検出
    person_detection(args)
    # shutil.rmtree(os.path.join(args.data_dir, "query_videos"))


if __name__ == "__main__":
    # args = parse_args()
    args = default_argument_parser().parse_args()
    args.gallery_features_csv = os.path.join(args.data_dir, "features.csv")
    args.gallery_action_features_npy = os.path.join(args.data_dir, "action_features.npy")
    args.gallery_person_features_npy = os.path.join(args.data_dir, "person_features.npy")

    if args.person_detection:
        retrieval_with_detection(args)
    else:
        retrieval(args)
        