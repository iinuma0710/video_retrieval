import os
import sys
import cv2
import csv
import glob
import argparse
import subprocess
import numpy as np
from darknet import *

sys.path.append("../")
from sort.sort import *

TV_STATIONS = ["bs1", "etv", "fuji", "net", "NHK", "ntv", "tbs", "tvtokyo"]


def convert_codec(input_video, output_video):
    subprocess.run(["ffmpeg", "-i", input_video, "-vcodec", "libx264", output_video])


class HumanDetectionAndTracking(object):
    """
    動作認識の前処理として人物領域の検出と追跡を行う．

    処理の流れ：
        1. Darknet (YOLO v4) による人物領域の検出を映像の各フレームに行う
        2. 1 で検出された人物を SORT で追跡する
        3. 検出・追跡された人物領域ごとに動画に書き出す

    TO DO：
        torch.Tensor の形式で動画を返せるようにする
    """

    def __init__(
        self,
        input_dir,  # ショットごとに分割した映像
        output_dir, # 人物ごとに検出を行なった映像
        input_video="", # 単一の映像から検出を行う場合にしてい
        config_file="../darknet/cfg/yolov4.cfg",
        weight_file="../darknet/weights/yolov4.weights",
        meta_file = "../darknet/cfg/coco.data",
        thresh=0.5,
        hier_thresh=0.5,
        nms=0.45
    ):
        self.output_dir = output_dir
        self.config_file = config_file
        self.weight_file = weight_file
        self.meta_file = meta_file
        self.thresh = thresh
        self.hier_thresh = hier_thresh
        self.nms = nms

        # 処理する映像のリストアップ
        if input_video != "":
            self.videos = [input_video]
        else:
            self.videos = glob.glob(os.path.join(input_dir, "*.mp4"))

        # ネットワークの読み込み
        self.net = self._load_network()
        # メタファイルの読み込み
        self.meta = self._load_meta_file()
        # 入力画像の雛形を作成
        self.darknet_image = make_image(network_width(self.net), network_height(self.net), 3)

        # 出力先のディレクトリがなければ作成
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # その他プライベート変数の初期化
        self.h, self.w, self.frame_num = [0] * 3
        self.fps = 0.0
        self.frames = []
        self.output_file = ""

    
    # ネットワークの読み込み
    def _load_network(self):
        print("Loading network ... \n (weight : {}, config : {})".format(self.weight_file, self.config_file))
        net = load_net_custom(self.config_file.encode("ascii"), self.weight_file.encode("ascii"), 0, 1)
        return net


    # メタファイルの読み込み
    def _load_meta_file(self):
        print("Loading meta data ... \n (meta_file : {})".format(self.meta_file))
        meta = load_meta(self.meta_file.encode("ascii"))
        return meta


    # 映像の読み込み
    def _read_video(self, video):
        cap = cv2.VideoCapture(video)
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                self.frames.append(frame)
            else:
                break

    
    # 映像の書き出し
    def _write_video(self, frames, w, h):
        fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
        mv = cv2.VideoWriter(self.output_file, fourcc, self.fps, (w, h))
        for f in frames:
            mv.write(f)
        mv.release()


    def _get_video_small_area(self, cx, cy, clip_w, clip_h, frame_indexes):
        # 幅の調整
        if cx + clip_w / 2 > self.w:
            min_x = int(max(0, self.w - clip_w))
            max_x = int(self.w)
        elif cx - clip_w / 2 < 0:
            min_x = 0
            max_x = int(min(self.w, clip_w))
        else:
            min_x = int(cx - clip_w / 2)
            max_x = int(cx + clip_w / 2)
        # 高さの調整
        if cy + clip_h / 2 > self.h:
            min_y = int(max(0, self.h - clip_h))
            max_y = int(self.h)
        elif cy - clip_h / 2 < 0:
            min_y = 0
            max_y = int(min(self.h, clip_h))
        else:
            min_y = int(cy - clip_h / 2)
            max_y = int(cy + clip_h / 2)

        print("small area (min_x, min_y, max_x, max_y) : ", (min_x, min_y, max_x, max_y))

        # 画像をリサイズして映像に書き出す
        frames_resized = [f[min_y:max_y, min_x:max_x] for f in self.frames]
        frames_resized = [frames_resized[i] for i in frame_indexes]
        self._write_video(frames_resized, max_x - min_x, max_y - min_y)


    def _get_video_big_area(self, cx, cy, person_w, person_h, clip_w, clip_h, frame_indexes):
        # 切り出し範囲の計算
        min_x = int(max(0, cx - person_w / 2))
        max_x = int(min(self.w, cx + person_w / 2))
        min_y = int(max(0, cy - person_h / 2))
        max_y = int(min(self.h, cy + person_h / 2))
        print("big area (min_x, min_y, max_x, max_y) : ", (min_x, min_y, max_x, max_y))

        # フレーム画像の切り出し
        frames_cliped = [f[min_y:max_y, min_x:max_x] for f in self.frames]
        frames_resized = [cv2.resize(f, (clip_w, clip_h)) for f in frames_cliped]
        frames_resized = [frames_resized[i] for i in frame_indexes]
        self._write_video(frames_resized, clip_w, clip_h)


    def get_video(self, bboxes, frame_indexes):
        # 人物の最大範囲を求める
        min_x, min_y, _, _, _ = bboxes.min(axis=0)
        _, _, max_x, max_y, _ = bboxes.max(axis=0)
        person_w, person_h = int(max_x - min_x), int(max_y - min_y)
        cx, cy = int((max_x + min_x) / 2), int((max_y + min_y) / 2)

        # 切り抜く領域を決める
        if person_w >= person_h:
            clip_w = int(person_w * 256 / person_h)
            clip_h = 256
        else:
            clip_h = int(person_h * 256 / person_w)
            clip_w = 256

        # 人物領域の短辺が 256 pixel 以下の場合 -> get_video_small_area
        if min(person_w, person_h) <= 256:
            self._get_video_small_area(cx, cy, clip_w, clip_h, frame_indexes)
        # 人物領域の短辺が 256 pixel より長いの場合 -> get_video_big_area
        else:
            self._get_video_big_area(cx, cy, person_w, person_h, clip_w, clip_h, frame_indexes)


    # フレーム画像から人物領域を検出する
    def _detect_human(self, frame):
        # フレーム画像の前処理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (network_width(self.net), network_height(self.net)), interpolation=cv2.INTER_LINEAR)
        copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        # 物体検出
        num = c_int(0)
        pnum = pointer(num)
        predict_image(self.net, self.darknet_image)
        letter_box = 0
        dets = get_network_boxes(self.net, self.darknet_image.w, self.darknet_image.h, self.thresh, self.hier_thresh, None, 0, pnum, letter_box)
        num = pnum[0]
        if self.nms:
            do_nms_sort(dets, num, self.meta.classes, self.nms)
    
        # 人物領域だけを取り出し，スケールを変換する
        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    nameTag = self.meta.names[i]
                    if nameTag == b'person':
                        # もとの画像スケールに戻す
                        cx = b.x * self.w / self.darknet_image.w
                        cy = b.y * self.h / self.darknet_image.h
                        w = b.w * self.w / self.darknet_image.w
                        h = b.h * self.h / self.darknet_image.h
                        # 左上と右下の座標に変換
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        res.append([x1, y1, x2, y2, dets[j].prob[i]])
        free_detections(dets, num)
    
        return res


    # 各映像について人物の検出とトラッキングを行う
    def detect_and_track_human(self):
        # for ループで各映像を処理
        for video in self.videos:
            print("Now processing : ", video)
            # 映像の読み出し
            self._read_video(video)
            # SORT の初期化
            sort_tracker = Sort()

            # フレームごとに人物領域を検出し逐次的にトラッキング
            frame_idx = 0
            frame_idx_dict = {}
            human_id_dict = {}
            for frame in self.frames:
                # 検出
                dets = self._detect_human(frame)
                # トラッキング
                dets = np.array(dets) if dets != [] else np.empty((0, 5))
                dets_track = sort_tracker.update(dets)
                # 検出された人物領域を整理
                for d in dets_track:
                    if int(d[4]) in human_id_dict:
                        human_id_dict[int(d[4])].append(d)
                        frame_idx_dict[int(d[4])].append(frame_idx)
                    else:
                        human_id_dict[int(d[4])] = [d]
                        frame_idx_dict[int(d[4])] = [frame_idx]
                frame_idx += 1
                
            # 動画に書き出す
            detected_videos = []
            for id in human_id_dict:
                # 短すぎる場合には飛ばす
                if len(frame_idx_dict[id]) < 64:
                    continue

                # 出力ファイル名を決める
                base_name = os.path.splitext(os.path.basename(video))[0]
                self.output_file = os.path.join(self.output_dir, base_name + "_id_{}_tmp.mp4".format(id))
                detected_video = os.path.join(self.output_dir, base_name + "_id_{}.mp4".format(id))
                # 書き出し
                track_array = np.array(human_id_dict[id])
                self.get_video(track_array, frame_idx_dict[id])
                # コーデックの変換
                convert_codec(self.output_file, detected_video)
                os.remove(self.output_file)
                detected_videos.append(detected_video)
        
        return detected_videos



# 引数の整理
def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--tv_station',
                        type=str,
                        choices=TV_STATIONS,
                        required=True,
                        help="TV Station"
                       )
    parser.add_argument('-y', '--year',
                        type=int,
                        default=2021,
                        help="Year"
                       )
    parser.add_argument('-m', '--month',
                        type=int,
                        default=2,
                        help="Year"
                       )
    parser.add_argument('-d', '--day',
                        type=int,
                        default=7,
                        help="Year"
                       )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    station = args.tv_station
    s_year = str(args.year)
    s_month = str(args.month).zfill(2)
    s_day = str(args.day).zfill(2)

    print("Processing {} programs of {}/{}/{}".format(station, s_day, s_month, s_year))

    input_dir="./data/shots/{}/{}/{}_{}_{}_04_56/".format(station, s_year, s_year, s_month, s_day)
    output_dir="./data/detected/{}/{}/{}_{}_{}_04_56/".format(station, s_year, s_year, s_month, s_day)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = HumanDetectionAndTracking(input_dir, output_dir)
    _ = detector.detect_and_track_human()