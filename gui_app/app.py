import os
import glob

from flask import request
from flask import Flask, render_template

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
		# 映像の一覧を取得
		if os.path.isdir(video_path):
			file_list = glob.glob(os.path.join(video_path, "*.mp4"))
		else:
			file_list = glob.glob(video_path + "*.mp4")
		# 共通のパスを取得し，static/ ディレクトリ以下にシンボリックリンクを貼る
		common_path = os.path.commonpath(file_list)
		os.symlink(common_path, static_query_video_dir)
		file_list = [f.replace(common_path, static_query_video_dir) for f in file_list]
		print(file_list)
		return render_template('query.html', file_list=file_list[:50])
	else:
		return render_template('query.html', file_list=[])


if __name__ == '__main__':
	app.run()