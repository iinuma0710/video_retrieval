import os
import csv
import argparse
import numpy as np


class NoFileExistsError(Exception):
    """ 必要なファイルがなかったときに呼び出されるエラー """


"""
検索を行うための関数群
"""
def retrieval_data(csv_file):
    # 検索のためのデータを読み出してくる
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        data_list = [[int(i), int(action_idx), int(person_idx)] for i, _, _, action_idx, person_idx, _, _, _ in reader]
    return data_list


def person_retrieval(query, galleries, query_person_fvs, gallery_person_fvs, args):
    query_idx, _, person_idx = query
    query_fv = query_person_fvs[person_idx]
    gallery_idx_dict = {item[2]: item for item in galleries}

    # コサイン類似度でランキングを作成 (特徴ベクトルは正規化済み)
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_person_fvs])
    ranking = np.argsort(-similarity)[:args.limit]

    # 検索結果から動作検索の対象を絞り込む
    refined_galleries = [gallery_idx_dict[idx] for idx in ranking]
    result = [query_idx] + [gallery_idx_dict[idx][0] for idx in ranking]
    return result, refined_galleries


def action_retrieval(query, galleries, query_action_fvs, gallery_action_fvs):
    query_idx, action_idx, _ = query
    query_fv = query_action_fvs[action_idx]
    gallery_fvs = gallery_action_fvs[[idx for _, idx, _ in galleries]]
    gallery_idx_dict = {idx: item for idx, item in enumerate(galleries)}

    # 2乗距離でランキングを作成 (特徴ベクトルは正規化済み)
    similarity = np.sum((gallery_fvs - query_fv) ** 2, axis=1)
    ranking = np.argsort(similarity)
    
    # 結果を返す
    result = [query_idx] + [gallery_idx_dict[idx][0] for idx in ranking]
    return result


def retrieval(args):
    # クエリとギャラリーの読み込み
    queries = retrieval_data(args.query_features_csv)
    galleries = retrieval_data(args.gallery_features_csv)

    # データの読み出し
    query_action_fvs = np.load(args.query_action_features_npy)
    query_person_fvs = np.load(args.query_person_features_npy)
    gallery_action_fvs = np.load(args.gallery_action_features_npy)
    gallery_person_fvs = np.load(args.gallery_person_features_npy)

    # クエリを1つずつ処理する
    person_rankings, action_rankings = [], []
    for query in queries:
        print("Now on processing Query : {}".format(query[0]))
        # 人物の検索
        person_ranking, refined_galleries = person_retrieval(query, galleries, query_person_fvs, gallery_person_fvs, args)
        person_rankings.append(person_ranking)
        # 動作の検索
        action_ranking = action_retrieval(query, refined_galleries, query_action_fvs, gallery_action_fvs)
        action_rankings.append(action_ranking)

    # 検索結果を保存し返す
    np.save(args.person_rankings_npy, np.array(person_rankings))
    np.save(args.action_rankings_npy, np.array(action_rankings))
    return np.array(action_rankings)


"""
評価を行うための関数群
"""
def validation_data(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        data_dict = {idx: [action_label, person_label] for idx, _, _, _, _, _, action_label, person_label in reader}
    return data_dict


def calc_ap(query_label, gallery_labels):
    correct_num = 0
    total_num = 0
    precision_sum = 0.0
    for gallery_label in gallery_labels:
        total_num += 1
        if query_label == gallery_label:
            correct_num += 1
            precision_sum += correct_num / total_num
        else:
            continue
    if correct_num == 0:
        ap = 0.0
    else:
        ap = precision_sum / correct_num
    return ap


def validation(args, rankings, mode=0):
    # 評価モード mode : 0 -> 動作, mode : 1 -> 人物
    print("Validation Mode : {}".format(mode))
    # クエリとギャラリーの読み込み
    queries = validation_data(args.query_features_csv)
    galleries = validation_data(args.gallery_features_csv)

    # 各クエリごとに AP を計算する
    label_ap_dict = {}
    for ranking in rankings:
        # 動作による検索の評価を行う準備
        query_idx, gallery_indexes = ranking[0], ranking[1:]
        query_action_label = queries[str(query_idx)][mode]
        gallery_action_labels = [galleries[str(gallery_idx)][mode] for gallery_idx in gallery_indexes]

        # AP を計算 
        ap = calc_ap(query_action_label, gallery_action_labels)
        if query_action_label in label_ap_dict:
            label_ap_dict[query_action_label].append(ap)
        else:
            label_ap_dict[query_action_label] = [ap]
        print("Query : {}, GT label : {}, AP : {}".format(query_idx, query_action_label, ap))

    # ラベルごとと全体の mAP を計算する
    map_list = []
    for key, val in label_ap_dict.items():
        map_per_label = np.mean(np.array(val))
        map_list.append(map_per_label)
        print("Query Label : {}, mAP : {}".format(key, map_per_label))
    map_total = np.mean(np.array(map_list))
    print("Total mAP : {}".format(map_total))


def parse_args():
    parser = argparse.ArgumentParser()

    # データディレクトリの指定
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help="Path to data directory"
                       )
    # 人物同定で何位まで検索を行うかを指定
    parser.add_argument('--limit',
                        type=int,
                        default=2000,
                        help="Limit of the ranking"
                       )
    # 検索のみを行う場合
    parser.add_argument('--only_retrieval',
                        action="store_true",
                        help="Validate retrieval results or not"
                       )
    # 動作による検索のみを行う場合(当面使わない)
    parser.add_argument('--only_action_retrieval',
                        action="store_true",
                        help="Validate retrieval results or not"
                       )
    # 人物による評価も行う場合(当面は使わない)
    parser.add_argument('--person_validation',
                        action="store_true",
                        help="Validate retrieval results or not"
                       )

    args = parser.parse_args()
    # ギャラリ関連のファイル
    args.gallery_features_csv = os.path.join(args.data_dir, "gallery/features.csv")
    args.gallery_action_features_npy = os.path.join(args.data_dir, "gallery/action_features.npy")
    args.gallery_person_features_npy = os.path.join(args.data_dir, "gallery/person_features.npy")
    # クエリ関連のファイル
    args.query_features_csv = os.path.join(args.data_dir, "query/features.csv")
    args.query_action_features_npy = os.path.join(args.data_dir, "query/action_features.npy")
    args.query_person_features_npy = os.path.join(args.data_dir, "query/person_features.npy")
    # 検索結果のランキングを保存しておくファイル
    args.person_rankings_npy = os.path.join(args.data_dir, "person_rankings.npy")
    args.action_rankings_npy = os.path.join(args.data_dir, "action_rankings.npy")

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

    if os.path.exists(args.action_rankings_npy):
        # rankings_npy があれば読み込み
        rankings = np.load(args.action_rankings_npy)
    else:
        # rankings_npy がなければ検索を行う
        rankings = retrieval(args)

    print(rankings.shape)

    if not args.only_retrieval:
        # 検索のみでなければ評価を行う
        validation(args, rankings, mode=0)
