import os
import csv
import numpy as np
import argparse
from multiprocessing import cpu_count, Pool


class NoFileExistsError(Exception):
    """ 必要なファイルがなかったときに呼び出されるエラー """


# 単一クエリに対してランキングを作成
def make_rank(arg):
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
    query_idx, query_fv, gallery_fvs, metric = arg

    # 類似度の計算
    # 値が小さくなるほど類似度が高くなるように，cos類似度は値を-1倍している
    if metric == "l2+norm" or metric == "l2":
        similarity = np.sum((gallery_fvs - query_fv) ** 2, axis=1)
    elif metric == "cosine":
        inner_product = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
        norm = np.array([np.linalg.norm(fv) * np.linalg.norm(query_fv) for fv in gallery_fvs])
        similarity = - inner_product / norm
    
    # 類似度スコア順にソートして limit で指定された上位のインデックスを返す
    ranking = np.argsort(similarity)
    print("Query : {} has been done.".format(query_idx))
    return query_idx, ranking


# 全クエリの検索を行う
def retrieval(gallery_file, query_file, save_file, metric="l2+norm"):
    # gallery と query の読み込み
    gallery_fvs = np.load(gallery_file)
    query_fvs = np.load(query_file)
    # metric が l2+norm なら，それぞれ正規化
    if metric == "l2+norm":
        gallery_fvs = gallery_fvs / np.array([np.sqrt(np.sum(gallery_fvs ** 2, axis=1))]).T
        query_fvs = query_fvs / np.array([np.sqrt(np.sum(query_fvs ** 2, axis=1))]).T
    
    # クエリごとの検索を並列で処理する
    args = [[query_idx, query_fvs[query_idx], gallery_fvs, metric] for query_idx in range(len(query_fvs))]
    cpu_num = max(len(args), int(cpu_count() * 0.8))
    pool = Pool(cpu_num)
    result = pool.map(make_rank, args)
    pool.close()
    rankings = np.array([r[1] for r in sorted(result)])

    # 検索結果を保存してランキングを返す
    np.save(save_file, rankings)
    return rankings


# 動作と人物による検索結果をマージする
def merge_ranking(action_ranking, person_ranking):
    pass


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
        action_ranking = retrieval(
            args.gallery_action_features_npy, 
            args.query_action_features_npy,
            args.action_ranking_npy,
            args.metric
        )
    else:
        print("Acrion retrieval already has done.")
    # 人物特徴量のファイルがあれば人物特徴量だけで検索をかけ，動作特徴量の検索結果にマージする
    if os.path.exists(args.gallery_person_features_npy) and os.path.exists(args.query_peron_features_npy):
        if not os.path.exists(args.action_ranking_npy):
            person_ranking = retrieval(
                args.gallery_person_features_npy, 
                args.query_person_features_npy,
                args.person_ranking_npy,
                args.metric
            )
            merge_ranking(action_ranking, person_ranking)
        else:
            print("Person retrieval already has done.")
    
    # validation フラグが True なら mAP の計算を行う
    if args.validation:
        pass

