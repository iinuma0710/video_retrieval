"""
次の一連の処理をコマンドラインから実行するためのアプリケーション
 - 映像をビデオショットに分割する
 - ビデオショットから人物映像を抽出する
 - 人物映像から特徴ベクトルを抽出する

引数の一覧
　・テレビ局と日付を指定して処理する場合
    --tv_station, -s テレビ局の指定 ["bs1", "etv", "fuji", "net", "NHK", "ntv", "tbs", "tvtokyo"]
    --year, -y       放送年の指定   2014以上の整数で指定
    --month, -m      放送月の指定   1〜12の整数で指定 (2014年は4月以降のみ)
    --day, -d        放送日の指定   1〜31の整数で指定
　
　・番組名を指定して処理する場合
    --programs  カンマ区切りで処理対象の番組名を指定する

　・処理したデータを格納するためのディレクトリ
    --data_dir      処理したデータを格納しておくディレクトリ
                    デフォルト(日付指定)：./data/
                    デフォルト(番組指定)：指定必須
    --recorded_dir  録画データを保存したディレクトリ
                    デフォルト(日付指定)：./data/recorded/<tv_station>/<year>/<year>_<month>_<day>_04_56/
                    デフォルト(番組指定)：指定必須
    --shots_dir     録画データをビデオショットごとに分割した映像を保存するディレクトリ
                    デフォルト(日付指定)：./data/shots/<tv_station>/<year>/<year>_<month>_<day>_04_56/
                    デフォルト(番組指定)：./<data_dir>/shots
    --detected_dir  ビデオショットから検出した人物映像を保存するディレクトリ
                    デフォルト(日付指定)：./data/detected/<tv_station>/<year>/<year>_<month>_<day>_04_56/
                    デフォルト(番組指定)：./<data_dir>/detected/
    --features_dir  抽出した特徴ベクトルの情報を出力するディレクトリ
                    デフォルト(日付指定)：./data/features/<tv_station>/<year>/<year>_<month>_<day>_04_56/
                    デフォルト(番組指定)：./<data_dir>/features/
    --tmp_dir       処理中の一時ファイルを保存するディレクトリ
                    デフォルト(日付指定)：./data/tmp/
                    デフォルト(番組指定)：./<data_dir>/tmp/

　・処理内容の指定
    --all        映像の分割，人物検出，特徴抽出を行う
    --split      映像の分割のみを行う
    --detection  人物の検出のみを行う
    --extraction 特徴抽出のみを行う

　・並列処理数の指定
    --cpu_num   映像の分割処理時の並列処理数を指定
"""


import os
import csv
import sys
import glob
import numpy as np
from multiprocessing import Pool

# 引数の取得
sys.path.append("../fast-reid")
from fastreid.engine import default_argument_parser
# 映像をショットに分割
from split_videos import split_video
# 人物映像の検出
from detection import HumanDetectionAndTracking
# 人物特徴の抽出
from person_feature import feature_extractor as person_feature_extractor
# 動作特徴の抽出
from action_feature import feature_extractor as action_feature_extractor


class DirectoryNotSpecifiedError(Exception):
    """ 指定の必要なディレクトリが設定されていない場合に呼び出されるエラー """


class DirectoryNotFinedError(Exception):
    """ 指定されたディレクトリが存在しない場合に呼び出されるエラー """


class PreviousProcessingNotFinished(Exception):
    """ 以前の処理がまだ完了していない場合に呼び出される """


def prepare_processing():
    # 引数の整理
    args = default_argument_parser().parse_args()

    if args.programs != "":
        # 番組名で絞り込みをかけている場合
        program_list = args.programs.split(',')
        print(program_list)
        # todo : 番組名での検索を導入
        if args.data_dir == "":
            raise DirectoryNotSpecifiedError("処理したデータを格納するディレクトリとして --data_dir を指定してください")
        sub_dir = ""
    elif args.recorded_dir != "":
        # 録画データを保存しているディレクトリを指定している場合
        if args.data_dir == "":
            raise DirectoryNotSpecifiedError("処理したデータを格納するディレクトリとして --data_dir を指定してください")
        sub_dir = ""
    else:
        # テレビ局と日時で処理対象を指定する場合
        args.data_dir = "./data"
        station = args.tv_station
        s_year = str(args.year)
        s_month = str(args.month).zfill(2)
        s_day = str(args.day).zfill(2)
        sub_dir = "{}/{}/{}_{}_{}_04_56/".format(station, s_year, s_year, s_month, s_day)

    # ディレクトリのパスを設定
    args.recorded_dir = os.path.join(args.data_dir, "recorded", sub_dir) if args.recorded_dir == "" else args.recorded_dir
    args.shots_dir = os.path.join(args.data_dir, "shots", sub_dir) if args.shots_dir == "" else args.shots_dir
    args.detected_dir = os.path.join(args.data_dir, "detected", sub_dir) if args.detected_dir == "" else args.detected_dir
    args.features_dir = os.path.join(args.data_dir, "features", sub_dir) if args.features_dir == "" else args.features_dir
    args.tmp_dir = os.path.join(args.data_dir, "tmp", sub_dir) if args.tmp_dir == "" else args.tmp_dir

    # 録画データが存在しなければエラーを返す
    if not os.path.exists(args.recorded_dir):
        raise DirectoryNotFinedError("録画データが存在しません")
    # ディレクトリが存在しなければ作成
    for directory in [args.shots_dir, args.detected_dir, args.features_dir, args.tmp_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 処理情報の表示
    print("録画データを格納したディレクトリ    => ", args.recorded_dir)
    print("ビデオショットを格納したディレクトリ=> ", args.shots_dir)
    print("人物映像を格納したディレクトリ      => ", args.detected_dir)
    print("特徴量の情報を格納したディレクトリ  => ", args.features_dir)
    print("一時ファイルを格納したディレクトリ  => ", args.tmp_dir)

    return args


def multiprocessing_split_video(arg):
    video, shots_dir = arg
    print("Now processing : ", video)
    split_video(video, shots_dir)


if __name__ == "__main__":
    # 処理の下準備
    args = prepare_processing()
    
    if args.all or args.split:
        # 映像の分割
        print("Now start spliting recorded videos...")
        pool = Pool(args.cpu_num)
        arg_list = [[rv, args.shots_dir] for rv in glob.glob(os.path.join(args.recorded_dir, "*.mp4"))]
        pool.map(multiprocessing_split_video, arg_list)
        
    if args.all or args.detection:
        if glob.glob(os.path.join(args.shots_dir, "*.mp4")) == []:
            raise PreviousProcessingNotFinished("先に映像の分割処理を行なってください")
        # 人物映像の検出
        print("Now start detect person videos from video shots...")
        detector = HumanDetectionAndTracking(args.shots_dir, args.detected_dir)
        _ = detector.detect_and_track_human()

    if args.all or args.extraction:
        if glob.glob(os.path.join(args.detected_dir, "*.mp4")) == []:
            raise PreviousProcessingNotFinished("先に人物映像の検出処理を行なってください")
        print("Now start feature extraction from person videos...")
        # 動作特徴の抽出
        action_feature_extractor(args.tmp_dir, args.detected_dir, args.features_dir)
        # 人物特徴の抽出
        person_feature_extractor(args.tmp_dir, args.detected_dir, args.features_dir, args)
    