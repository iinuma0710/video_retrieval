# 物体検出と動作認識を用いた映像検索
物体検出(YOLO v4)と動作認識(SlowFast Networks)を用いて映像の特徴量を抽出し映像検索を行う．

## ディレクトリの構成
```
/video_retrieval  
  |- /darknet  
  |- /slowfast  
  |- /sort  
  |- extract_people.py
```

## YOLO v4
映像中の人物領域のみを切り出す．
実装は [Darknet](https://github.com/AlexeyAB/darknet) を用いる．

### セットアップ
[ここ](https://github.com/AlexeyAB/darknet)から Darknet のソースコードをクローンしてくる．
```bash
$ git clone https://github.com/AlexeyAB/darknet
```

Makefile を次のように変更する．
Darknet は動画ファイルの物体検出にも対応しているが，動作認識に必要なフレームのみを抽出して物体検出をかけるので，
CUDNN と CUDNN_HALF は切っておく．
また，Python から叩けるようにするため LIBSO を1にしておく．
```Makefile
GPU=1
CUDNN=0
CUDNN_HALF=0
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0 # ZED SDK 3.0 and above
ZED_CAMERA_v2_8=0 # ZED SDK 2.X
```

## SlowFast Networks
各人物領域の動作について特徴量を抽出する．
実装は [Facebook Research の公式実装](https://github.com/facebookresearch/SlowFast)を用いる．
