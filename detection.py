"""
Darknet (YOLO v4) による人物領域の検出を行う
    入力：映像のフレーム列 (バッチ処理)
    出力：人物領域のバウンディングボックス

Darknet の詳細は https://github.com/AlexeyAB/darknet を参照
学習済みモデルは https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT からダウンロード
"""

import os
import cv2
import numpy as np
from ctypes import *


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



# libdarknet.so を読み込む
hasGPU = True
lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)


class Darknet(object):
    """
        Darknet を用いて人物の検出を行うクラス

        Attributes
        ----------
            config_file : ネットワークの設定ファイル
            weight_file : 学習済みネットワークの重みを保存したファイル
            meta_file   : データセットの設定ファイル

        Returns
        -------
        detections : 検出結果
    """

    def __init__(self, config_file, weight_file, meta_file, batch_size=32, thresh= 0.25, hier_thresh=.5, nms=.45):
        self.config_file = config_file
        self.weight_file = weight_file
        self.meta_file = meta_file
        self.batch_size = batch_size
        self.thresh = thresh
        self.hier_thresh = hier_thresh
        self.nms = nms

        # ネットワークの読み込み
        self.net = load_net_custom(self.config_file.encode('utf-8'), self.weight_file.encode('utf-8'), 0, batch_size)
        self.meta = load_meta(self.meta_file.encode('utf-8'))

    
    def _resize_images(self, images, net_height, net_width, c):
        # 画像をネットワークの入力サイズに合うようにリサイズする
        imgs = []
        for img in images:
            # Darknet の入力サイズに合わせて画像をリサイズ
            custom_img = cv2.resize(img, (net_width, net_height), interpolation=cv2.INTER_NEAREST)
            # [チャネル, 横, 縦] の順番に転置する
            custom_img = custom_img.transpose(2, 0, 1)
            # 画像をリストに追加
            imgs.append(custom_img)

        # 画像のリストを ndarray に変換する
        arr = np.concatenate(imgs, axis=0)
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        # C のコードで扱えるようにテンソルを変換する
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(net_width, net_height, c, data)
        
        return im

    
    def _get_bboxes(self, dets, num):
        # 個別の画像の検出結果について整理する
        boxes, scores, classes = [], [], []
        for i in range(num):
            det = dets[i]
            score = -1
            label = None

            # print(det.bbox.x, det.bbox.y, det.bbox.h, det.bbox.w)
            for c in range(det.classes):
                p = det.prob[c]
                # 確率が最大のクラスを選択する
                if p > score:
                    score = p
                    label = c

            # バウンディングボックスのクラス，スコア，座標を計算
            if score > self.thresh:
                box = det.bbox
                left, top, right, bottom = map(int,(box.x - box.w / 2, box.y - box.h / 2, box.x + box.w / 2, box.y + box.h / 2))
                boxes.append((top, left, bottom, right))
                scores.append(score)
                classes.append(label)

        return boxes, scores, classes


    def _parse_detections(self, batch_dets):
        # 検出結果を整理してバウンディングボックスを取得する
        batch_boxes, batch_scores, batch_classes = [], [], []
        for idx in range(self.batch_size):
            num = batch_dets[idx].num
            dets = batch_dets[idx].dets
            if self.nms:
                do_nms_obj(dets, num, self.meta.classes, self.nms)
            
            boxes, scores, classes = self._get_bboxes(dets, num)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_classes.append(classes)

        return batch_boxes, batch_scores, batch_classes


    def detect(self, images):
        # images はフレーム画像のバッチ，画像はすべて同じサイズでなければならない
        # images は [横, 縦, チャネル数, バッチサイズ] の4次元テンソル
        
        # 画像バッチの前処理
        pred_height, pred_width, c = images[0].shape
        net_width, net_height = (network_width(self.net), network_height(self.net))
        imgs = self._resize_images(images, net_height, net_width, c)

        # 検出
        dets = network_predict_batch(self.net, imgs, self.batch_size, pred_width, pred_height, self.thresh, self.hier_thresh, None, 0, 0)

        # 検出結果の整理
        bboxes, scores, classes = self._parse_detections(dets)

        # メモリの開放
        free_batch_detections(dets, self.batch_size)

        return bboxes, scores, classes



if __name__ == "__main__":
    import pprint

    detector = Darknet(
        config_file="./darknet/cfg/yolov4.cfg",
        weight_file="./darknet/weights/yolov4.weights",
        meta_file="./darknet/cfg/coco.data",
        batch_size=3
    )
    img_samples = ['./darknet/data/person.jpg', './darknet/data/person.jpg', './darknet/data/person.jpg']
    image_list = [cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB) for k in img_samples]
    pprint.pprint(detector.detect(image_list))

