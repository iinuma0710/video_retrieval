import os
import csv
import numpy as np


def load_data(csv_file):
    with open(csv_file, "r") as fp:
        reader = csv.reader(fp, delimiter=' ')
        feature_idx_dict = {}
        action_label_dict = {}
        for r in reader:
            feature_idx = int(r[3])
            action_label = int(r[6])
            feature_idx_dict[feature_idx] = action_label
            if action_label != 99:
                if action_label in action_label_dict:
                    action_label_dict[action_label].append(feature_idx)
                else:
                    action_label_dict[action_label] = [feature_idx]
            else:
                continue
    
    return feature_idx_dict, action_label_dict


def retrieval(query_idx, features, metric="l2+norm", limit=100):
    # 類似度の計算
    # 値が小さくなるほど類似度が高くなるように，cos類似度は値を-1倍している
    if metric == "l2+norm":
        normalized = action_features / np.array([np.sqrt(np.sum(action_features ** 2, axis=1))]).T
        similarity = np.sum((normalized - normalized[query_idx]) ** 2, axis=1)
    elif metric == "l2":
        similarity = np.sum((normalized - normalized[query_idx]) ** 2, axis=1)
    elif metric == "cosine":
        inner_product = np.array([np.dot(fv, features[query_idx]) for fv in features])
        norm = np.array([np.linalg.norm(fv) * np.linalg.norm(features[query_idx]) for fv in features])
        similarity = - inner_product / norm
    else:
        print("metric の値が不正です")
        return
    
    # 類似度スコア順にソートして limit で指定された上位のインデックスを返す
    ranking = np.argsort(similarity)
    return ranking[:limit]


def calc_ap(query_idx, rank, feature_idx_dict):
    gt = feature_idx_dict[query_idx]
    correct_num = 0
    total_num = 0
    precision_sum = 0.0
    for idx in rank:
        total_num += 1
        if gt == feature_idx_dict[idx]:
            # 正解ならば正解数と precision を加算していく
            correct_num += 1
            precision_sum += correct_num / total_num
        else:
            # 不正解なら何もしない
            continue
    # average precision を計算
    ap = precision_sum / correct_num
    return ap


if __name__ == "__main__":
    action_features = np.load("data/eastenders/action_features.npy")
    feature_idx_dict, action_label_dict = load_data("data/eastenders/features.csv")

    rank = retrieval(98, action_features)
    ap = calc_ap(98, rank, feature_idx_dict)
    print(ap)