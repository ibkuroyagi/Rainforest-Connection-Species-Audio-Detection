# Rainforest Connection Species Audio Detection
### Automate the detection of bird and frog species in a tropical soundscape

## 締め切り
February 17, 2021 - Final submission deadline.

## introduction
ホームディレクトリに.kaggleディレクトリが作成されている前提で作成します。 
ない場合は、こちら[https://www.currypurin.com/entry/2018/kaggle-api](https://www.currypurin.com/entry/2018/kaggle-api)を参照してください。
```
# リポジトリのクローン
git clone https://github.com/ibkuroyagi/Rainforest-Connection-Species-Audio-Detection.git
# 環境構築
cd Rainforest-Connection-Species-Audio-Detection/tools
make
```
<details><summary>slurm用にヒアドキュメントを使用する場合</summary><div>

```
cd Rainforest-Connection-Species-Audio-Detection/tools
sbatch -c 4 -w million2 << EOF
#!/bin/bash
make
EOF
```

</div></details>


## コンペの主題は何?
少量のデータセットかつ外部ノイズの多い音からAudio-taggingをする
ただし、複数の種類が存在する可能性がある
- noisy-labelの活用
- ダイアライゼーションを使用
- augmentationによるnoise対策
- 特徴抽出部分を事前学習
## 注意すること
- スペクトログラムのスケールでlogを取ると高周波成分が失われるので注意
- CVの切り方をtestと同じ分布にする

## アイデア
- segment-wise predoction(種によって鳴いている時間の長さが異なる)
- wav2vec
- SpecAug
- ダイアライゼーションの出力を代入(ラベルがオーバーラップしている)
- 無音区間予測で精度改善?

## 決定事項
- 初手のCVの切り方はiterative-stratificationを用いる
- 推論に使うデータをいい感じにクロップするためにダイアライゼーションコードを作成する


<details><summary>kaggle日記</summary><div>

- 11/29(日)
    - 今日やったこと
        * リポジトリ作成&コンペの理解
    - 次回やること
        * 手元環境でのEDAとstage1の作成
- 12/9(水)
    - 今日やったこと
        * 手元環境でのEDAとstage1の作成
    - 次回やること
        * 手元環境でのEDAとstage1の作成
- 12/10(木)
    - 今日やったこと
        * preprocess完成
    - 次回やること
        * models, datasets, lossesの作成
- 12/11(金)
    - 今日やったこと
        * models, datasets, lossesの作成
    - 次回やること
        * trainer, bin/sed_trainの作成
- 12/12(土)
    - 今日やったこと
        * trainer, bin/sed_trainの作成
    - 次回やること
        * trainer, bin/sed_trainの作成
- 12/13(日)
    - 今日やったこと
        * trainer, bin/sed_trainの作成
    - 次回やること
        * clip ratioとlrを調整v003~v004
- 12/14(月)
    - 今日やったこと
        * clip ratioとlrを調整v003~v004
    - 次回やること
        * tensorboardをいい感じに作成
- 12/15(火)
    - 今日やったこと
        * tensorboardをいい感じに作成
    - 次回やること
        * 推論コードを作成
- 12/16(水)
    - 今日やったこと
        * 推論コードを作成、run.shを編集
    - 次回やること
        * 推論を実行
- 12/17(木)
    - 今日やったこと
        * 推論を実行
    - 次回やること
        * 推論結果提出, 推論時の後処理を分析
- 12/18(金)
    - 今日やったこと
        * 推論結果提出, 推論時の後処理を分析
    - 次回やること
        * 
</div></details>
