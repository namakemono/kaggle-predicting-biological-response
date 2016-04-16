## 概要
- Kaggleの過去問Predicting a Biological Responseを説いた結果です．
- https://www.kaggle.com/c/bioresponse
- 38位LogLoss 0.38106 でした．

Stacked Generalizationと呼ばれる方法を利用しています

### 実行方法
https://www.kaggle.com/c/bioresponse/data
からtrain.csv, test.csvをダウンロードしてinputsフォルダに入れて下記コマンドを実行すると，outputsに提出用ファイルが生成されます．

```
cd bio
python train.csv
```
