# Triplet Margin Loss を導入する
表現学習ができるように Triplet Margin Loss を導入する．

## pytorch-metric-learning の導入
Triplet Loss は公式にサポートがないので，サードパーティ製のライブラリを使う．
詳細は [pytorch-metric-learning のサイト](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/)を参照．

インストールは，
```bash
$ pip install pytorch-metric-learning
```