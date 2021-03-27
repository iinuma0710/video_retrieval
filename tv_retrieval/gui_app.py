import os
import csv
import sys
import glob
import random
import shutil
import numpy as np
# GUI アプリ向け
from flask import request
from flask import Flask, render_template
# 引数の取得
sys.path.append("../fast-reid")
from fastreid.engine import default_argument_parser
# 人物映像の検出
from detection import HumanDetectionAndTracking
# 人物特徴の抽出
from person_feature import feature_extractor_from_video as person_feature_extractor
# 動作特徴の抽出
from action_feature import feature_extractor_from_video as action_feature_extractor

app = Flask(__name__)


# コサイン類似度による検索
def cosine_similarity(query_fv, gallery_fvs, person_ret_num):
    similarity = np.array([np.dot(fv, query_fv) for fv in gallery_fvs])
    ranking = np.argsort(-similarity)[:person_ret_num]
    return ranking


# L2ノルムによる検索
def l2_similarity(query_fv, gallery_fvs, action_ret_num):
    similarity = np.sum((gallery_fvs - query_fv) ** 2, axis=1)
    ranking = np.argsort(similarity)[:action_ret_num]
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
    person_query_fv = person_feature_extractor(args.query_video, args)
    person_gallery_fvs = np.array([person_fvs[int(d[1])] for d in data_list])
    person_result = cosine_similarity(person_query_fv, person_gallery_fvs, args.person_ret_num)
    refined_data_list = [data_list[i] for i in person_result]

    # 動作の検索
    print("Retrieving with action features ...")
    action_query_fv = action_feature_extractor(args.query_video)
    action_gallery_fvs = np.array([action_fvs[int(d[0])] for d in refined_data_list])
    action_result = l2_similarity(action_query_fv, action_gallery_fvs, args.action_ret_num)
    result_data_list = [refined_data_list[i] for i in action_result]
    action_res_list = [d[2] for d in result_data_list]

    return action_res_list


# 人物の検索のみを行う関数
def retrieval_person(args):
    # データの読み込み
    print("Reading retrieval data ...")
    person_fvs = np.load(args.gallery_person_features_npy)
    with open(args.gallery_features_csv, "r") as f:
        data_list = [row for row in csv.reader(f)]
        
    # 人物の検索
    print("Retrieving with person features ...")
    person_query_fv = person_feature_extractor(args.query_video, args)
    person_gallery_fvs = np.array([person_fvs[int(d[1])] for d in data_list])
    person_result = cosine_similarity(person_query_fv, person_gallery_fvs, args.person_ret_num)
    refined_data_list = [data_list[i] for i in person_result]
    person_res_list = [d[2] for d in refined_data_list]
	
    return person_res_list[:args.action_ret_num]


# 動作の検索のみを行う関数
def retrieval_action(args):
    # データの読み込み
    print("Reading retrieval data ...")
    action_fvs = np.load(args.gallery_action_features_npy)
    with open(args.gallery_features_csv, "r") as f:
        data_list = [row for row in csv.reader(f)]
        
    # 動作の検索
    print("Retrieving with action features ...")
    action_query_fv = action_feature_extractor(args.query_video)
    action_gallery_fvs = np.array([action_fvs[int(d[0])] for d in data_list])
    action_result = l2_similarity(action_query_fv, action_gallery_fvs, args.action_ret_num)
    result_data_list = [data_list[i] for i in action_result]
    action_res_list = [d[2] for d in result_data_list]

    return action_res_list


# 人物検索と動作検索をフュージョンする
def retrieval_person_action_fusion(args):
	# データの読み込み
    print("Reading retrieval data for fusion ...")
    action_fvs = np.load(args.gallery_action_features_npy)
    person_fvs = np.load(args.gallery_person_features_npy)
    with open(args.gallery_features_csv, "r") as f:
        data_list = [row for row in csv.reader(f)]
        
    # 人物の検索
    print("Retrieving with person features ...")
    person_query_fv = person_feature_extractor(args.query_video, args)
    person_gallery_fvs = np.array([person_fvs[int(d[1])] for d in data_list])
    person_sim = np.array([np.dot(fv, person_query_fv) for fv in person_gallery_fvs])

    # 動作の検索
    print("Retrieving with action features ...")
    action_query_fv = action_feature_extractor(args.query_video)
    action_gallery_fvs = np.array([action_fvs[int(d[0])] for d in data_list])
    action_sim = np.array([np.dot(fv, action_query_fv) for fv in action_gallery_fvs])

	# 検索結果をフュージョン
    fused_sim = person_sim + action_sim
    ranking = np.argsort(-fused_sim)[:args.action_ret_num]
    res = [data_list[i][2] for i in ranking]

    return res


@app.route('/retrieval', methods=['POST', 'GET'])
def retrieval():
	# 以前のシンボリックリンクがないか確認
	static_gallery_video_dir = "static/gallery_videos"
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
		if request.form.get('person_only'):
			file_list = retrieval_person(args)
		elif request.form.get('action_only'):
			file_list = retrieval_action(args)
		elif request.form.get('fusion'):
			file_list = retrieval_person_action_fusion(args)
		else:
			file_list = retrieval_person_action(args)

		# ファイルのパスを絶対パスに変換
		new_file_list = []
		for f in file_list:
			if not os.path.isabs(f):
				new_file_list.append(os.path.abspath(f))
			else:
				new_file_list.append(f)
		file_list = new_file_list
		
		# 共通のパスを取得し，static/ ディレクトリ以下にシンボリックリンクを貼る
		common_path = os.path.commonpath(file_list)
		os.symlink(common_path, static_gallery_video_dir)
		display_file_list = [f.replace(common_path, static_gallery_video_dir) for f in file_list]
		file_num = len(file_list)
		action_id_list = ["action_" + str(i + 1) for i in range(file_num)]
		person_id_list = ["person_" + str(i + 1) for i in range(file_num)]

		return render_template(
			'retrieval.html',
			file_list=file_list,
			display_file_list=display_file_list,
			file_num=file_num,
			action_id_list=action_id_list,
			person_id_list=person_id_list
		)
	else:
		return render_template(
			'retrieval.html',
			file_list=[],
			display_file_list=[],
			file_num=0,
			action_id_list=[],
			person_id_list=[]
		)


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
		if not os.path.isabs(video_path):
			video_path = os.path.abspath(video_path)
		
		# 映像が指定されたときは人物の検出を行う
		if os.path.isfile(video_path):
			save_dir = request.form.get("save_dir")
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			detector = HumanDetectionAndTracking(input_dir=None, output_dir=save_dir, input_video=video_path)
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
		return render_template('query.html', file_list=file_list, display_file_list=display_file_list, file_num=len(file_list))
	else:
		return render_template('query.html', file_list=[], display_file_list=[], file_num=0)


# テレビ映像の人物映像の中からランダムにクエリの候補を表示するページ
@app.route('/random_query', methods=['POST', 'GET'])
def random_query():
	if request.method == 'POST':
		# 表示件数の取得
		limit = int(request.form.get('radio'))
		# 処理済みのテレビの人物映像の一覧を取得する
		query_candidate_list = glob.glob("static/detected/*/*/*/*.mp4")
		query_list = random.sample(query_candidate_list, limit)
		full_path_list = [os.path.join(os.getcwd(), "data", q[7:]) for q in query_list]
		return render_template('random_query.html', file_list=full_path_list, display_file_list=query_list, file_num=len(query_list))
	else:
		return render_template('random_query.html', file_list=[], display_file_list=[], file_num=0)


@app.route('/query_retrieval', methods=['POST', 'GET'])
def query_retrieval():
	# 以前のシンボリックリンクがないか確認
	static_query_video_dir = "static/query_videos"
	if os.path.exists(static_query_video_dir):
		os.unlink(static_query_video_dir)
	static_gallery_video_dir = "static/gallery_videos"
	if os.path.exists(static_gallery_video_dir):
		os.unlink(static_gallery_video_dir)
	
	if request.method == 'POST':
		# 検索に必要な情報を取得する
		args = default_argument_parser().parse_args()
		args.query_video = request.form.get('radio')
		if request.form.get('retrieve_from') == "trecvid":
			args.data_dir = "/home/iinuma/per610a/video_retrieval/data/retrieval_data/"
		else:
			args.data_dir = "./data/features/2021_02_07_bs1_etv_nhk/"
		args.person_ret_num = int(request.form.get("person_ret_num"))
		args.action_ret_num = int(request.form.get("action_ret_num"))
		args.gallery_features_csv = os.path.join(args.data_dir, "features.csv")
		args.gallery_action_features_npy = os.path.join(args.data_dir, "action_features.npy")
		args.gallery_person_features_npy = os.path.join(args.data_dir, "person_features.npy")
		
		# 検索を行う
		if request.form.get('person_only'):
			file_list = retrieval_person(args)
		elif request.form.get('action_only'):
			file_list = retrieval_action(args)
		elif request.form.get('fusion'):
			file_list = retrieval_person_action_fusion(args)
		else:
			file_list = retrieval_person_action(args)

		# ファイルのパスを絶対パスに変換
		new_file_list = []
		for f in file_list:
			if not os.path.isabs(f):
				new_file_list.append(os.path.abspath(f))
			else:
				new_file_list.append(f)
		file_list = new_file_list
		
		# 共通のパスを取得し，static/ ディレクトリ以下にシンボリックリンクを貼る
		common_path = os.path.commonpath(file_list)
		os.symlink(common_path, static_gallery_video_dir)
		display_file_list = [f.replace(common_path, static_gallery_video_dir) for f in file_list]
		file_num = len(file_list)
		action_id_list = ["action_" + str(i + 1) for i in range(file_num)]
		person_id_list = ["person_" + str(i + 1) for i in range(file_num)]

		# クエリ映像の表示用
		display_query_video = os.path.join("./static/", os.path.basename(args.query_video))
		if not os.path.exists(display_query_video):
			shutil.copyfile(args.query_video, display_query_video)

		return render_template(
			'query_retrieval.html',
			query_candidate_list=[display_query_video],
			display_query_list=[display_query_video],
			query_candidate_num=1,
			file_list=file_list,
			display_file_list=display_file_list,
			file_num=file_num,
			action_id_list=action_id_list,
			person_id_list=person_id_list
		)
	else:
		if request.args.get('query_select', '') == "trecvid":
			query_candidate_list = glob.glob("/net/per610a/export/das18a/satoh-lab/share/datasets/eastenders/video_detected/*.mp4")
		else:
			query_candidate_list = glob.glob("/home/iinuma/per610a/video_retrieval/tv_retrieval/data/detected/*/*/*/*.mp4")
		
		query_candidate_list = random.sample(query_candidate_list, 5)
		common_path = os.path.commonpath(query_candidate_list)
		os.symlink(common_path, static_query_video_dir)
		display_query_list = [f.replace(common_path, static_query_video_dir) for f in query_candidate_list]

		return render_template(
			'query_retrieval.html',
			query_candidate_list=query_candidate_list,
			display_query_list=display_query_list,
			query_candidate_num=len(query_candidate_list),
			file_list=[],
			display_file_list=[],
			file_num=0,
			action_id_list=[],
			person_id_list=[]
		)


if __name__ == '__main__':
	app.run()