"""
Darknet (YOLO v4) による人物領域の検出を行う
    入力：映像のフレーム列 (バッチ処理)
    出力：人物領域のバウンディングボックス

Darknet の詳細は https://github.com/AlexeyAB/darknet を参照
学習済みモデルは https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT からダウンロード
"""

import cv2
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
                # res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    # res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


def detect_video(net, meta, video):
    # 映像ファイルの読み込み
    cap = cv2.VideoCapture(video)
    # 入力画像の雛形
    darknet_image = make_image(network_width(net), network_height(net), 3)
    # SORT によるトラッキングの初期化
    sort_tracker = Sort()

    frame_cnt = 0
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

            # people = []
            # for det in dets:
            #     if det[0] == b'person':
            #         people.append(det)

            # print(frame_cnt, people)
            print(frame_cnt, dets_track)
            frame_cnt += 1
        else:
            break


if __name__ == "__main__":
    net = load_network()
    meta = load_meta_file()
    video = "/net/per610a/export/das18a/satoh-lab/share/datasets/kinetics600/video/train/walking_the_dog/0sL5rRoMgLs_000015_000025.mp4"
    detect_video(net, meta, video)