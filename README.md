# 物体検出と動作認識を用いた映像検索
物体検出(YOLO v4)と動作認識(SlowFast Networks)を用いて映像の特徴量を抽出し映像検索を行う．

## ディレクトリの構成
```
/video_retrieval  
  |- /darknet  
  |- /slowfast  
  |- /sort
  |- clipping.py
  |- darknet.py 
  |- detection.py
  |- 
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


## SORT
検出したそれぞれの人物領域についてトラッキングを行う．
SORT はバウンディングボックスの座標とスコアから，カルマン・フィルタなどを用いてトラッキングを行う．  
必要なパッケージをインストールすればセットアップ完了．
```bash
$ cd sort
$ pip install -r requirements.txt
```

## YOLO v4 と SORT の動作確認
detection.py を実行してみてエラーが出なければOK.
```bash
$ python detection.py
```

## SlowFast Networks
各人物領域の動作について特徴量を抽出する．
実装は [Facebook Research の公式実装](https://github.com/facebookresearch/SlowFast)を用いる．

セットアップは [インストールガイド](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md) を参考に行う．

1. 必要なパッケージをインストール  
  PyTorch と TorchVision は [PyTorch のサポートページ](https://pytorch.org/)を参考に，CUDAのバージョンなどを揃える．
```bash
# numpy
$ pip install numpy 
# fvcore
$ pip install 'git+https://github.com/facebookresearch/fvcore'
# simplejson
$ pip install simplejson
$ conda install av -c conda-forge
# psutil
$ pip install psutil
# OpenCV
pip install opencv-python
# tensorboard
pip install tensorboard
# Detectron 2
$ pip install -U torch torchvision cython
$ pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
$ git clone https://github.com/facebookresearch/detectron2 detectron2_repo
$ pip install -e detectron2_repo
```

2. Path の設定
  video_retrieval/slowfast/ のフルパスを ```PYTHONPATH``` に追加する．
```.bashrc
export PYTHONPATH="/path/to/video_retrieval/slowfast:$PYTHONPATH"
```

3. ビルドする
```bash
$ cd slowfast
$ python setup.py build develop
```

セットアップが完了したら，テストを実行して動作確認する．  
(※ **データセットやモデルはこのリポジトリ内にはないので，あとから準備します．**)
```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```
