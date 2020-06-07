# 物体検出と動作認識を用いた映像検索
物体検出(YOLO v4)と動作認識(SlowFast Networks)を用いて映像の特徴量を抽出し映像検索を行う．

## ディレクトリの構成
```
/video_retrieval  
  |- /darknet  
  |- /slowfast  
  |- /sort  
  |- detection.py
```

## YOLO v4
映像中の人物領域のみを切り出す．
実装は [Darknet](https://github.com/AlexeyAB/darknet) を用いる．

Makefile を次のように変更する．
Darknet は動画ファイルの物体検出にも対応しているが，動作認識に必要なフレームのみを抽出して物体検出をかけるので，
CUDNN と CUDNN_HALF はここでは使わないが．必要に応じて使用する．
また，Python から叩けるようにするため LIBSO を1にしておく．
```bash
$ cd darknet
$ vim Makefile
  
  GPU=1
  CUDNN=0
  CUDNN_HALF=0
  OPENCV=1
  AVX=0
  OPENMP=0
  LIBSO=1
  ZED_CAMERA=0 # ZED SDK 3.0 and above
  ZED_CAMERA_v2_8=0 # ZED SDK 2.X
  
$ make
```
続いて，[Google Drive](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) から YOLO v4 の学習済みモデルをダウンロードし，```./darknet/weights/yolov4.weights``` に保存しておく．
これで，セットアップは完了．
detection.py を実行してみてエラーが出なければOK.
```bash
$ cd ..
$ python detection.py
```

## SlowFast Networks
各人物領域の動作について特徴量を抽出する．
実装は [Facebook Research の公式実装](https://github.com/facebookresearch/SlowFast)を用いる．
