import os
import csv
import sys
import glob
import numpy as np
# GUI アプリ向け
from flask import request
from flask import Flask, render_template
# 引数の取得
sys.path.append("./fast-reid")
from fastreid.engine import default_argument_parser
# 人物映像の検出
from detection_2 import HumanDetectionAndTracking
# 人物特徴の抽出
from person_feature import feature_extractor as person_feature_extractor
# 動作特徴の抽出
from action_feature import feature_extractor as action_feature_extractor

app = Flask(__name__)


# コサイン類似度による検索
def cosine_similarity(query_fv, gallery_fvs, person_ret_num):
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
    ranking = np.argsort(-similarity)[:person_ret_num]
    return ranking


# L2ノルムによる検索
def l2_similarity(query_fv, gallery_fvs, action_ret_num):
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
    ranking = np.argsort(-similarity)[:action_ret_num]
    return ranking


# 検索を行う関数
def retrieval_person_action(args):
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
    person_result = cosine_similarity(person_query_fv, person_gallery_fvs, args.person_ret_num)
    refined_data_list = [data_list[i] for i in person_result]

    # 動作の検索
    print("Retrieving with action features ...")
    action_query_fv = action_feature_extractor([args.query_video])[0]
    action_gallery_fvs = np.array([action_fvs[int(d[0])] for d in refined_data_list])
    action_result = l2_similarity(action_query_fv, action_gallery_fvs, args.action_ret_num)
    result_data_list = [refined_data_list[i] for i in action_result]
    
    # 検索結果の出力
    res_list = [d[2] for d in result_data_list]
    return res_list


@app.route('/retrieval', methods=['POST', 'GET'])
def retrieval():
	# 以前のシンボリックリンクがないか確認
	static_gallery_video_dir = "static/gallery_videos"
	print(os.path.exists(static_gallery_video_dir))
	if os.path.exists(static_gallery_video_dir):
		os.unlink(static_gallery_video_dir)
	
	if request.method == 'POST':
		# 検索に必要な情報を取得する
		args = default_argument_parser().parse_args()
		args.query_video = request.form.get("query_video")
		args.data_dir = request.form.get("gallery_dir")
		args.person_ret_num = int(request.form.get("person_ret_num"))
		args.action_ret_num = int(request.form.get("action_ret_num"))
		args.gallery_features_csv = os.path.join(args.data_dir, "features.csv")
		args.gallery_action_features_npy = os.path.join(args.data_dir, "action_features.npy")
		args.gallery_person_features_npy = os.path.join(args.data_dir, "person_features.npy")
		# 検索を行う
		file_list = retrieval_person_action(args)
		# 共通のパスを取得し，static/ ディレクトリ以下にシンボリックリンクを貼る
		common_path = os.path.commonpath(file_list)
		os.symlink(common_path, static_gallery_video_dir)
		display_file_list = [f.replace(common_path, static_gallery_video_dir) for f in file_list]

		return render_template('retrieval.html', file_list=file_list, display_file_list=display_file_list, file_num=len(file_list))
	else:
		return render_template('retrieval.html', file_list=[], display_file_list=[], file_num=0)


# クエリの動画を探すためのページ
@app.route('/query', methods=['POST', 'GET'])
def query():
	# 以前のシンボリックリンクがないか確認
	static_query_video_dir = "static/query_videos"
	print(os.path.exists(static_query_video_dir))
	if os.path.exists(static_query_video_dir):
		os.unlink(static_query_video_dir)

	if request.method == 'POST':
		# HTMLからディレクトリのパスを取得
		video_path = request.form.get("video_path")
		
		# 映像が指定されたときは人物の検出を行う
		if os.path.isfile(video_path):
			save_dir = request.form.get("save_dir")
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			detector = HumanDetectionAndTracking(input_video=video_path, output_dir=save_dir)
			file_list = detector.detect_and_track_human()
		# ディレクトリが指定された場合にはディレクトリ内の全ての映像を列挙
		elif os.path.isdir(video_path):
			file_list = glob.glob(os.path.join(video_path, "*.mp4"))
		# それ以外の場合には指定されたパスを含む映像を列挙
		else:
			file_list = glob.glob(video_path + "*.mp4")

		# 人物映像が検出できなければ何も表示しない
		if file_list == []:
			return render_template('query.html', file_list=[], display_file_list=[], file_num=0)

		# 共通のパスを取得し，static/ ディレクトリ以下にシンボリックリンクを貼る
		common_path = os.path.commonpath(file_list)
		os.symlink(common_path, static_query_video_dir)
		display_file_list = [f.replace(common_path, static_query_video_dir) for f in file_list]
		# 映像が多い場合には50件だけ表示
		# todo : ページネーションで全件表示
		return render_template('query.html', file_list=file_list[:50], display_file_list=display_file_list[:50], file_num=len(file_list))
	else:
		return render_template('query.html', file_list=[], display_file_list=[], file_num=0)


if __name__ == '__main__':
	app.run()