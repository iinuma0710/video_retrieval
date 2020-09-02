import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count, Pool


class NoFileExistsError(Exception):
    """ 必要なファイルがなかったときに呼び出されるエラー """


# 単一クエリに対してランキングを作成
def make_rank(query_idx):
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
    query_fv = query_fvs[query_idx]

    # 類似度の計算
    # 値が小さくなるほど類似度が高くなるように，cos類似度は値を-1倍している
    if args.metric == "l2+norm" or args.metric == "l2":
        similarity = np.sum((gallery_fvs - query_fv) ** 2, axis=1)
    elif args.metric == "cosine":
        inner_product = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
        norm = np.array([np.linalg.norm(fv) * np.linalg.norm(query_fv) for fv in gallery_fvs])
        similarity = - inner_product / norm
    
    # 類似度スコア順にソートして limit で指定された上位のインデックスを返す
    ranking = np.argsort(similarity)
    # print("Query : {} has been done.".format(query_idx))
    return query_idx, ranking


# 全クエリの検索を行う
def retrieval(gallery_file, query_file, save_file):
    # gallery と query の読み込み
    global gallery_fvs
    gallery_fvs = np.load(gallery_file)
    global query_fvs
    query_fvs = np.load(query_file)
    # metric が l2+norm なら，それぞれ正規化
    if args.metric == "l2+norm":
        gallery_fvs = gallery_fvs / np.array([np.sqrt(np.sum(gallery_fvs ** 2, axis=1))]).T
        query_fvs = query_fvs / np.array([np.sqrt(np.sum(query_fvs ** 2, axis=1))]).T
    
    # クエリごとの検索を並列で処理する
    parallel_args = list(range(len(query_fvs)))
    cpu_num = min(len(parallel_args), int(cpu_count() * 0.8))
    print("Start retrieval : {} queries, {} CPU cores".format(len(parallel_args), cpu_num))
    with Pool(cpu_num) as pool:
        imap = pool.imap(make_rank, parallel_args)
        result = list(tqdm(imap, total=len(parallel_args)))
        rankings = np.array([r[1] for r in sorted(result)])
    print("Retrieval has been finished. Result is saved in", save_file)

    # 検索結果を保存してランキングを返す
    np.save(save_file, rankings)
    return rankings


# 動作と人物による検索結果をマージする
def merge_ranking(action_rankings, person_rankings):
    return action_rankings


# Average Precision の計算
def calc_ap(gt_label, ranking, gallery_dict):
    correct_num = 0
    total_num = 0
    precision_sum = 0.0
    for r in ranking:
        total_num += 1
        if gt_label == gallery_dict[r]:
            correct_num += 1
            precision_sum += correct_num / total_num
        else:
            continue
    ap = precision_sum / correct_num
    return ap


# calc_ap 関数を並列実行するためのラッパー
def wrapper_calc_ap(arg):
    gt_label, ranking, gallery_dict = arg
    ap = calc_ap(gt_label, ranking, gallery_dict)
    return ap


# mean Average Precision の計算
def calc_map(rankings, gallery_dict, query_dict):
    """
    gallery_dict : { feature_vector_idx : label }
    query_dict : { lebel : [feature_vector_idx] }
    """
    ap_dict = {}
    mean_ap = 0.0
    for label in query_dict:
        # 同じラベルの映像については並列に AP を計算する
        parallel_args = [[label, rankings[idx], gallery_dict] for idx in query_dict[label]]
        cpu_num = min(len(parallel_args), int(cpu_count() * 0.8))
        pool = Pool(cpu_num)
        result = np.array(pool.map(make_rank, parallel_args))
        pool.close()

        # ラベルごとの AP
        label_ap = np.mean(result)
        ap_dict[label] = label_ap
        mean_ap += label_ap / len(query_dict)

    return mean_ap, ap_dict


def load_gallery_data(csv_file):
    action_gallery_dict = {}
    person_gallery_dict = {}
    with open(csv_file, "r") as fp:
        reader = csv.reader(fp, delimiter=' ')
        for r in reader:
            if r[3] != '' and r[6] != '':
                action_gallery_dict[r[3]] = r[6]
            if r[4] != '' and r[7] != '':
                action_gallery_dict[r[4]] = r[7]
    
    return action_gallery_dict, person_gallery_dict


def load_query_data(csv_file):
    action_query_dict = {}
    person_query_dict = {}
    with open(csv_file, "r") as fp:
        reader = csv.reader(fp, delimiter=' ')
        for r in reader:
            if r[3] != '' and r[6] != '':
                if r[6] in action_query_dict:
                    action_query_dict[r[6]].append(r[3])
                else:
                    action_query_dict[r[6]] = [r[3]]
            if r[4] != '' and r[7] != '':
                if r[4] in action_query_dict:
                    action_query_dict[r[7]].append(r[4])
                else:
                    action_query_dict[r[7]] = [r[4]]

    return action_query_dict, person_query_dict


# 引数の整理
def parse_args():
    parser = argparse.ArgumentParser()

    # データディレクトリの指定
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help="Path to data directory"
                       )
    # 特徴ベクトルの距離指標
    parser.add_argument('--metric',
                        type=str,
                        choices=["l2+norm", "l2", "cos"],
                        default="l2+norm",
                        help="Distance betweeen feature vectors"
                       )
    # 上位何位まで検索を行うかを指定
    parser.add_argument('--limit',
                        type=int,
                        default=100,
                        help="Limit of the ranking"
                       )
    # 検索結果の評価を行うかどうかを指定
    parser.add_argument('--validation',
                        action="store_true",
                        help="Validate retrieval results or not"
                       )

    args = parser.parse_args()
    args.gallery_features_csv = os.path.join(args.data_dir, "gallery/features.csv")
    args.gallery_action_features_npy = os.path.join(args.data_dir, "gallery/action_features.npy")
    args.gallery_person_features_npy = os.path.join(args.data_dir, "gallery/person_features.npy")
    args.query_features_csv = os.path.join(args.data_dir, "query/features.csv")
    args.query_action_features_npy = os.path.join(args.data_dir, "query/action_features.npy")
    args.query_person_features_npy = os.path.join(args.data_dir, "query/person_features.npy")
    args.action_ranking_npy = os.path.join(args.data_dir, "query/action_ranking.npy")
    args.person_ranking_npy = os.path.join(args.data_dir, "query/person_ranking.npy")
    args.total_ranking_npy = os.path.join(args.data_dir, "query/total_ranking.npy")

    # 最低限必要なファイルがあるかどうかを確認する
    if not os.path.exists(args.gallery_features_csv):
        raise NoFileExistsError("{} does not exist.".format(args.gallery_features_csv))
    if not os.path.exists(args.query_features_csv):
        raise NoFileExistsError("{} does not exist.".format(args.query_features_csv))
    if not os.path.exists(args.gallery_action_features_npy):
        raise NoFileExistsError("{} does not exist.".format(args.gallery_action_features_npy))
    if not os.path.exists(args.query_action_features_npy):
        raise NoFileExistsError("{} does not exist.".format(args.query_action_features_npy))

    return args


if __name__ == "__main__":
    args = parse_args()

    # 動作特徴量による検索
    if not os.path.exists(args.action_ranking_npy):
        action_rankings = retrieval(
            args.gallery_action_features_npy, 
            args.query_action_features_npy,
            args.action_ranking_npy
        )
    else:
        action_rankings = np.load(args.action_ranking_npy)
        print("Acrion retrieval already has done.")
    # 人物特徴量のファイルがあれば人物特徴量だけで検索をかけ，動作特徴量の検索結果にマージする
    if os.path.exists(args.gallery_person_features_npy) and os.path.exists(args.query_peron_features_npy):
        if not os.path.exists(args.person_ranking_npy):
            person_rankings = retrieval(
                args.gallery_person_features_npy, 
                args.query_person_features_npy,
                args.person_ranking_npy
            )
            rankings = merge_ranking(action_rankings, person_rankings)
        else:
            rankings = np.load(args.total_ranking_npy)
            print("Person retrieval already has done.")
    else:
        rankings = action_rankings
    
    # validation フラグが True なら mAP の計算を行う
    if args.validation:
        gallery_dict, _ = load_gallery_data(args.gallery_features_csv)
        query_dict, _ = load_query_data(args.query_features_csv)

        mean_ap, _ = calc_map(rankings, gallery_dict, query_dict)



