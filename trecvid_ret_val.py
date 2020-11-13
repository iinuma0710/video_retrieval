import os
import csv
import argparse
import numpy as np
import xml.etree.ElementTree as ET

class NoFileExistsError(Exception):
    """ 必要なファイルがなかったときに呼び出されるエラー """


# 単一クエリに対してランキングを作成
def make_rank(query_idx, gallery_indexes):
    """
    並列化のため args は配列として渡す
    args の中身
        query_idx : int
            クエリのインデックス
        query_fv : ndarray([2304,])
            クエリの特徴ベクトル
        gallery_fvs : ndarray([gallery_num, 2304])
            検索対象の特徴ベクトル
        metric : str (l2+norm or l2 or cosine)
            特徴ベクトル間の距離指標
    """
    query_fv = args.query_fvs[int(query_idx)]
    target_gallery_fvs = args.gallery_fvs[gallery_indexes]

    # 類似度の計算
    similarity = np.sum((target_gallery_fvs - query_fv) ** 2, axis=1)
    # 類似度スコア順にソートして limit で指定された上位のインデックスを返す
    ranking = np.argsort(similarity)
    # print("Query : {} has been done.".format(query_idx))
    return ranking


# XML ファイルをパースして TRECVID INS のクエリを読み込む
def read_query(args):
    action_query_dict = {}
    with open(args.query_features_csv, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for _, _, _, query_idx, _, _, action_label, _ in reader:
            if action_label in action_query_dict:
                action_query_dict[action_label].append(query_idx)
            else:
                action_query_dict[action_label] = [query_idx]

    tree = ET.parse(args.query_xml)
    root = tree.getroot()
    query_dict = {}
    for child in root:
        action_id = args.action_label_dict[child.attrib["action"]]
        person_id = args.person_label_dict[child.attrib["person"]]
        if action_id not in action_query_dict:
            break
        for query_idx in action_query_dict[action_id]:
            if action_id in query_dict:
                query_dict[action_id].append([query_idx, person_id])
            else:
                query_dict[action_id] = [[query_idx, person_id]]
    
    return query_dict


# CSV ファイルから gallery の情報を読み出す
def read_gallery(args):
    """
    gallery_dict = {person_label : [ [gallery_idx, action_label], ..., [gallery_idx, action_label] ] }
    """
    gallery_dict = {}
    with open(args.gallery_features_csv, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for _, _, _, gallery_idx, _, _, action_label, person_label in reader:
            if person_label in gallery_dict:
                gallery_dict[person_label].append([gallery_idx, action_label])
            else:
                gallery_dict[person_label] = [[gallery_idx, action_label]]

    return gallery_dict


# クエリごとにAP(Average Precision)を計算
def calc_ap(gt_label, ranking_labels):
    correct_num = 0
    total_num = 0
    precision_sum = 0.0
    for label in ranking_labels:
        total_num += 1
        if gt_label == label:
            correct_num += 1
            precision_sum += correct_num / total_num
        else:
            continue
    
    ap = precision_sum / correct_num
    return ap


# 検索と評価を行う関数
def ret_val():
    total_queries = 0
    result_list = []
    map_sum = 0.0
    
    for query_action_id, queries in args.query_dict.items():
        ap_sum = 0.0
        
        for query_idx, query_person_id in queries:
            total_queries += 1
            # 特徴量情報とラベル情報の分離
            gallery_indexes = np.array([int(gallery_idx) for gallery_idx, _ in args.gallery_dict[query_person_id]])
            gallery_labels = np.array([int(gallery_label) for _, gallery_label in args.gallery_dict[query_person_id]])
            # 検索 (ランキングの作成)
            ranking = make_rank(query_idx, gallery_indexes)
            res = []
            ranking_new = []
            for r in ranking:
                i = gallery_indexes[r]
                if i not in res:
                    res.append(i)
                    ranking_new.append(r)
            ranking = np.array(ranking_new)
            # Average Precision の計算
            ap = calc_ap(int(query_action_id), gallery_labels[ranking])
            print("Action Label : {}, Person Label : {}, Average Precision : {}".format(query_action_id, query_person_id, ap))
            ap_sum += ap
            # 結果を記録
            result_list.append(np.insert(gallery_indexes[ranking[:int(args.limit)]], 0, int(query_idx)))
        
        action_map = ap_sum / len(queries)
        print("Action Label : {}, mAP/action : {}".format(query_action_id, action_map))
        map_sum += ap_sum
    
    total_map = map_sum / total_queries
    print("mAP: ", total_map)

    return np.array(result_list)


# 引数の整理
def parse_args():
    parser = argparse.ArgumentParser()

    # データディレクトリの指定
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help="Path to data directory"
                       )
    # 上位何位まで検索を行うかを指定
    parser.add_argument('--limit',
                        type=int,
                        default=100,
                        help="Limit of the ranking"
                       )

    args = parser.parse_args()
    args.gallery_features_csv = os.path.join(args.data_dir, "gallery/features.csv")
    args.gallery_features_npy = os.path.join(args.data_dir, "gallery/action_features.npy")
    args.query_xml = os.path.join(args.data_dir, "topics/ins.auto.topics.2019.xml")
    args.query_features_csv = os.path.join(args.data_dir, "query/features.csv")
    args.query_features_npy = os.path.join(args.data_dir, "query/action_features.npy")
    args.results_npy = os.path.join(args.data_dir, "results.npy")

    # 最低限必要なファイルがあるかどうかを確認する
    if not os.path.exists(args.gallery_features_csv):
        raise NoFileExistsError("{} does not exist.".format(args.gallery_features_csv))
    if not os.path.exists(args.query_features_csv):
        raise NoFileExistsError("{} does not exist.".format(args.query_features_csv))
    if not os.path.exists(args.gallery_features_npy):
        raise NoFileExistsError("{} does not exist.".format(args.gallery_features_npy))
    if not os.path.exists(args.query_features_npy):
        raise NoFileExistsError("{} does not exist.".format(args.query_features_npy))

    # ラベル情報を読み込む
    print("Loading action and person label information...")
    with open(os.path.join(args.data_dir, "query/action_labels.csv"), "r") as f:
        reader = csv.reader(f, delimiter=" ")
        args.action_label_dict = {r[0]: r[1] for r in reader}

    with open(os.path.join(args.data_dir, "query/person_labels.csv"), "r") as f:
        reader = csv.reader(f, delimiter=" ")
        args.person_label_dict = {r[0].capitalize(): r[1] for r in reader}

    # 特徴ベクトルを読み込んで正規化 (各ベクトルの長さを1にする)
    print("Loading feature vectors...")
    query_fvs_tmp = np.load(args.query_features_npy)
    args.query_fvs = query_fvs_tmp / np.array([np.sqrt(np.sum(query_fvs_tmp ** 2, axis=1))]).T
    gallery_fvs_tmp = np.load(args.gallery_features_npy)
    args.gallery_fvs = gallery_fvs_tmp / np.array([np.sqrt(np.sum(gallery_fvs_tmp ** 2, axis=1))]).T

    # クエリと検索対象の情報を読み込み
    print("Loading feature vectors' information...")
    args.query_dict = read_query(args)
    args.gallery_dict = read_gallery(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    results = ret_val()
    np.save(args.results_npy, results)