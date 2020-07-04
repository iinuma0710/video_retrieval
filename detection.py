"""
Darknet (YOLO v4) による人物領域の検出を行う
    入力：映像のフレーム列 (バッチ処理)
    出力：人物領域のバウンディングボックス

Darknet の詳細は https://github.com/AlexeyAB/darknet を参照
学習済みモデルは https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT からダウンロード
"""

import cv2
import pprint
import numpy as np
from darknet import *
from sort.sort import *


# ネットワークの読み込み
def load_network(config_file = "./darknet/cfg/yolov4.cfg", weight_file = "./darknet/weights/yolov4.weights"):
    print("Loading network ... \n (weight : {}, config : {})".format(weight_file, config_file))
    net = load_net_custom(config_file.encode("ascii"), weight_file.encode("ascii"), 0, 1)
    return net

# メタデータの読み込み
def load_meta_file(meta_file = "./darknet/cfg/coco.data"):
    print("Loading meta data ... \n (meta_file : {})".format(meta_file))
    meta = load_meta(meta_file.encode("ascii"))
    return meta


# 各フレーム画像からの検出
def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    letter_box = 0
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                nameTag = meta.names[i]
                if nameTag == b'person':
                    x1 = b.x - b.w / 2
                    y1 = b.y - b.h / 2
                    x2 = b.x + b.w / 2
                    y2 = b.y + b.h / 2
                    res.append([x1, y1, x2, y2, dets[j].prob[i]])
    free_detections(dets, num)
    
    return res


def read_video(video):
    # 映像ファイルの読み込み
    cap = cv2.VideoCapture(video)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    return h, w, frame_num, fps, frames


def write_video(video, frames, fps, min_x, min_y, max_x, max_y):
    frames_resized = [f[min_y:max_y, min_x:max_x] for f in frames]

    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    mv = cv2.VideoWriter(video, fourcc, fps, (max_x - min_x, max_y - min_y))
    for f in frames_resized:
        mv.write(f)
    mv.release()


def get_video_small_area(input_video, output_video, cx, cy, clip_w, clip_h):
    original_h, original_w, frame_num, fps, frames = read_video(input_video)

    # 幅の調整
    if cx + clip_w / 2 > original_w:
        min_x = int(max(0, original_w - clip_w))
        max_x = int(original_w)
    elif cx - clip_w / 2 < 0:
        min_x = 0
        max_x = int(min(original_w, clip_w))
    else:
        min_x = int(cx - clip_w / 2)
        max_x = int(cx + clip_w / 2)
    # 高さの調整
    if cy + clip_h / 2 > original_h:
        min_y = int(max(0, original_h - clip_h))
        max_y = int(original_h)
    elif ch - clip_h / 2 < 0:
        min_y = 0
        max_y = int(min(original_h, clip_h))
    else:
        min_y = int(cy - clip_h / 2)
        max_y = int(cy + clip_h / 2)

    print(output_video, "(min_x, min_y, max_x, max_y) : ", (min_x, min_y, max_x, max_y))
    write_video(output_video, frames, fps, min_x, min_y, max_x, max_y)


def get_video_big_area(input_video, output_video, cx, cy, person_w, person_h, clip_w, clip_h):
    pass


def get_video(input_video, output_video, bboxes):
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
        get_video_small_area(input_video, output_video, cx, cy, clip_w, clip_h)
    # 人物領域の短辺が 256 pixel より長いの場合 -> get_video_big_area
    else:
        get_video_big_area(input_video, output_video, cx, cy, person_w, person_h,)


def detect_video(net, meta, video):
    # 映像ファイルの読み込み
    cap = cv2.VideoCapture(video)
    # 入力画像の雛形
    darknet_image = make_image(network_width(net), network_height(net), 3)
    # SORT によるトラッキングの初期化
    sort_tracker = Sort()

    # frame_cnt = 0
    person_id_dict = {}
    while True:
        # フレーム画像のキャプチャ
        ret, frame = cap.read()
        if ret:
            # フレーム画像の前処理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (network_width(net), network_height(net)), interpolation=cv2.INTER_LINEAR)
            copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            # 検出
            dets = detect_image(net, meta, darknet_image, thresh=0.25)
            # トラッキング
            dets = np.array(dets) if dets != [] else np.empty((0, 5))
            dets_track = sort_tracker.update(dets)
            # 検出された人物領域を整理
            for d in dets_track:
                if int(d[4]) in person_id_dict:
                    person_id_dict[int(d[4])].append(d)
                else:
                    person_id_dict[int(d[4])] = [d]
        else:
            break
    
    # 人物領域の範囲
    cnt = 1
    for id in person_id_dict:
        track_array = np.array(person_id_dict[id])
        get_video(video, "test_{}.mp4".format(cnt), track_array)
        cnt += 1


if __name__ == "__main__":
    net = load_network()
    meta = load_meta_file()
    # video = "/net/per610a/export/das18a/satoh-lab/share/datasets/kinetics600/video/train/walking_the_dog/0sL5rRoMgLs_000015_000025.mp4"
    video = "/net/per610a/export/das18a/satoh-lab/share/datasets/kinetics600/video/train/dancing_ballet/3GewP3XbJHE_000050_000060.mp4"
    detect_video(net, meta, video)