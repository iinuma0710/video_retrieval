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
def retrieval():
    global rankings

    start = 0
    interval = 10000
    aps = []

    # クエリごとの検索を1万件ずつ並列で処理する
    query_indexes = list(range(len(query_fvs)))
    cpu_num = min(len(query_indexes), int(cpu_count() * 0.8))
    print("Start retrieval : {} queries, {} CPU cores".format(len(query_indexes), cpu_num))

    while start < len(query_indexes):
        # 並列処理の設定
        parallel_args = query_indexes[start:start+interval]
        
        # ランキングの作成
        with Pool(cpu_num) as pool:
            imap = pool.imap(make_rank, parallel_args)
            result = list(tqdm(imap, total=len(parallel_args)))
            new_rankings = np.array([r[1] for r in sorted(result)])
        print("Retrieval has been finished. {} / {}".format(start + len(parallel_args), len(query_indexes)))

        # ランキングの保存
        if os.path.exists(args.ranking_file):
            prev_rankings = np.load(args.ranking_file)
            rankings = np.concatenate([prev_rankings, new_rankings])
        else:
            rankings = new_rankings
        np.save(args.ranking_file, rankings)

        # validation フラグが true なら mAP を計算する
        if args.validation:
            with Pool(cpu_num) as pool:
                imap = pool.imap(calc_ap, parallel_args)
                result = list(tqdm(imap, total=len(parallel_args)))
                aps += result
            print("mAP : {} ({} queries)".format(np.mean(np.array(aps)), start + len(parallel_args)))
        
        start += interval

    return rankings


# Average Precision の計算
def calc_ap(query_idx):
    gt_label = query_dict[query_idx]
    ranking = rankings[query_idx][:args.limit]

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


# mean Average Precision の計算
def calc_map(query_label_dict):
    """
    query_label_dict : { lebel : [feature_vector_idx] }
    """
    ap_dict = {}
    aps = []
    
    for label in query_label_dict:
        # 同じラベルの映像については並列に AP を計算する
        parallel_args = query_label_dict[label]
        cpu_num = min(len(parallel_args), int(cpu_count() * 0.8))
        with Pool(cpu_num) as pool:
            imap = pool.imap(calc_ap, parallel_args)
            result = list(tqdm(imap, total=len(parallel_args)))
            aps += result

        # ラベルごとの AP
        label_ap = np.mean(np.array(result))
        ap_dict[label] = label_ap
        print("{} class AP : {} ({} queries)".format(label, label_ap, len(parallel_args)))

    mean_ap = np.mean(np.array(aps))
    print("All class mAP : {} ({} queries)".format(label_ap, len(aps)))

    return mean_ap, ap_dict


def load_index_data(csv_file):
    action_index_dict = {}
    with open(csv_file, "r") as fp:
        reader = csv.reader(fp, delimiter=' ')
        for r in reader:
            if r[3] != '' and r[6] != '':
                action_index_dict[int(r[3])] = int(r[6])
    
    return action_index_dict


def load_label_data(csv_file):
    action_label_dict = {}
    with open(csv_file, "r") as fp:
        reader = csv.reader(fp, delimiter=' ')
        for r in reader:
            if r[3] != '' and r[6] != '':
                if r[6] in action_label_dict:
                    action_label_dict[int(r[6])].append(int(r[3]))
                else:
                    action_label_dict[int(r[6])] = [int(r[3])]

    return action_label_dict


# 引数の整理
def parse_args():
    parser = argparse.ArgumentParser()

    # データディレクトリの指定
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help="Path to data directory"
                       )
    # 検索結果を保存するファイルの指定
    parser.add_argument('--ranking_file',
                        type=str,
                        required=True,
                        help="Path to the file saving result ranking"
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
    # 評価のみを行う場合
    parser.add_argument('--validation_only',
                        action="store_true",
                        help="Validate retrieval results or not"
                       )

    args = parser.parse_args()
    args.gallery_features_csv = os.path.join(args.data_dir, "gallery/features.csv")
    args.gallery_action_features_npy = os.path.join(args.data_dir, "gallery/action_features.npy")
    args.query_features_csv = os.path.join(args.data_dir, "query/features.csv")
    args.query_action_features_npy = os.path.join(args.data_dir, "query/action_features.npy")

    # 最低限必要なファイルがあるかどうかを確認する
    if not os.path.exists(args.gallery_features_csv):
        raise NoFileExistsError("{} does not exist.".format(args.gallery_features_csv))
    if not os.path.exists(args.query_features_csv):
        raise NoFileExistsError("{} does not exist.".format(args.query_features_csv))
    if not os.path.exists(args.gallery_action_features_npy):
        raise NoFileExistsError("{} does not exist.".format(args.gallery_action_features_npy))
    if not os.path.exists(args.query_action_features_npy):
        raise NoFileExistsError("{} does not exist.".format(args.query_action_features_npy))

    # 評価のみを行う指定で，ranking_flle が存在しないときはエラー
    if args.validation_only and not os.path.exists(args.ranking_file):
        raise NoFileExistsError("{} does not exist.".format(args.ranking_file))

    return args


if __name__ == "__main__":
    # 引数の整理，ファイルの設定
    args = parse_args()
    
    # validation_only フラグが True なら mAP の計算のみを行う
    if args.validation_only or os.path.exists(args.ranking_file):
        gallery_dict = load_index_data(args.gallery_features_csv)
        query_dict = load_index_data(args.query_features_csv)
        rankings = np.load(args.ranking_file)

        print("Start validation.")
        query_label_dict = load_label_data(args.query_features_csv)
        mean_ap, ap_dict = calc_map(query_label_dict)
    else:
        # gallery と query の読み込み
        gallery_fvs = np.load(args.gallery_action_features_npy)
        query_fvs = np.load(args.query_action_features_npy)

        # metric が l2+norm なら，それぞれ正規化
        if args.metric == "l2+norm":
            gallery_fvs = gallery_fvs / np.array([np.sqrt(np.sum(gallery_fvs ** 2, axis=1))]).T
            query_fvs = query_fvs / np.array([np.sqrt(np.sum(query_fvs ** 2, axis=1))]).T
        
        # 評価のために action の index と GT label の辞書を作る
        if args.validation:
            gallery_dict = load_index_data(args.gallery_features_csv)
            query_dict = load_index_data(args.query_features_csv)

        # 動作特徴量による検索
        action_rankings = retrieval()



