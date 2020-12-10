import os
import glob
# GUI アプリ向け
from flask import request
from flask import Flask, render_template
# 人物映像の検出
from detection_2 import HumanDetectionAndTracking

app = Flask(__name__)


@app.route('/retrieval')
def retrieval():
	return render_template('retrieval.html')


# クエリの動画を探すためのページ
@app.route('/query', methods=['POST', 'GET'])
def query():
	# 以前のシンボリックリンクがないか確認
	static_query_video_dir = "static/query_videos"
	print(os.path.exists(static_query_video_dir))
	if os.path.exists(static_query_video_dir):
		os.unlink(static_query_video_dir)

	# 描画の処理
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