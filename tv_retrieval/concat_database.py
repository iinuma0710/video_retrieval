import os
import csv
import sys
import numpy as np


args = sys.argv

# 新しく書き出すファイル
new_features_dir = args[1]
if not os.path.exists(new_features_dir):
    os.makedirs(new_features_dir)
new_action_features_npy = os.path.join(new_features_dir, "action_features.npy")
new_person_features_npy = os.path.join(new_features_dir, "person_features.npy")
new_features_csv = os.path.join(new_features_dir, "features.csv")

# 各ファイルを読み込んで新しい配列に詰め替えていく
new_action_feature_idx = 0
new_person_feature_idx = 0
new_action_features = []
new_person_features = []
new_features = []
for features_dir in args[2:]:
    # 結合対象のデータファイルのパス
    action_features_npy = os.path.join(features_dir, "action_features.npy")
    person_features_npy = os.path.join(features_dir, "person_features.npy")
    features_csv = os.path.join(features_dir, "features.csv")
    
    # 結合対象のデータを読み込み
    action_features = np.load(action_features_npy)
    person_features = np.load(person_features_npy)
    with open(features_csv, "r") as f:
        features_list = [r for r in csv.reader(f)]

    # 結合作業
    for action_idx, person_idx, video_path in features_list:
        print(video_path)
        new_action_features.append(action_features[int(action_idx)])
        new_person_features.append(person_features[int(person_idx)])
        new_features.append([new_action_feature_idx, new_person_feature_idx, video_path])
        new_action_feature_idx += 1
        new_person_feature_idx += 1

# 書き出し
np.save(new_action_features_npy, np.array(new_action_features))
np.save(new_person_features_npy, np.array(new_person_features))
with open(new_features_csv, "w") as f:
    csv.writer(f).writerows(new_features)
        