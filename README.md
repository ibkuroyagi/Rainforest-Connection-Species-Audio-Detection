# Rainforest Connection Species Audio Detection
### Automate the detection of bird and frog species in a tropical soundscape

## 締め切り
February 17, 2021 - Final submission deadline.

## CV vs PL
スプレッドシートにCVとPLの関係を記録する
[https://docs.google.com/spreadsheets/d/10_kbyXUlYpSEzLPpDA3JL_s4R1IsHNfuKhdA5gp6L4c/edit?usp=sharing](https://docs.google.com/spreadsheets/d/10_kbyXUlYpSEzLPpDA3JL_s4R1IsHNfuKhdA5gp6L4c/edit?usp=sharing)

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
- ~~segment-wise predoction(種によって鳴いている時間の長さが異なる)~~
- ~~CenterLoss~~
- wav2vec
- ~~SpecAug~~
- ダイアライゼーションの出力を代入(ラベルがオーバーラップしている)
- 無音区間予測で精度改善?
- ~~ResNexst50をtorchvisionから重みをインポートしてファインチューニング(埋め込み層を明示的に作成)~~
- EENDをconformerで実装
- Mixupを実装
- Augmentationがデータセット内でできるようにwave形式の入力を受け取るようにdataset, collater_fcに追記
- ~~TimeStretchが0.9, 1.1のmatrix_tpを作成~~

## 決定事項
- 初手のCVの切り方はiterative-stratificationを用いる
- 推論に使うデータをいい感じにクロップするためにダイアライゼーションコードを作成する
- augmentationは[dcase2020-task4-1st](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Miyazaki_108.pdf)にならって,mixup, time-shiftを導入
- time-shiftはASR分野では0.9, 1.1倍にすると実験的に良いという報告(espnet)からその数値を採用

## 実験結果からの気づき
- v004(num_mels=64, max_frame=1024)のスコアが著しく低いことから周波数成分の分解濃が重要だと思われる.
- 全てのモデルにおいて訓練データのlossが下がり切っているので,タスクが簡単過ぎる可能性が高い.より難易度の高い問題設定にする必要がある.
- どのモデルもだいたい500stepsあたりで過学習になり始めるので、そこの調整が重要そう
- 今回のmetricであるlwlrapは順位のみが影響するので、sumよりbinaryを用いて確実に1を出力するモデルを作ることが重要
    - 下手にノイズの鳥の声で高い確率が出力されてしまい順位を誤ると大きな痛手になる
- mel64hop512は総じて低いスコア
    - つまり、周波数方向の情報が重要な要素となっているので、その活用が効くと推測
- embeddingから直接clipwise_outputを求めると精度劇的に悪化する -> 時間成分を考慮しない(外部環境音のみで識別する)モデルになるため推論時に超悪さする
- lrは1.0e-4のオーダーが良く効く
- CenterLossの正則化を強くしすぎると過学習を起こしてしまい、悪化する -> noiseクラスを許す(random=True)とinferenceとの差がなくなるので効果的
- 学習がゆっくり進むようにmax_frame2048, batch_size128がほどよい
- PANNs事前学習の重みはかなり効果的
- 
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
        * v003, num_mels: 128, hop_size: 512 -> 1024 (window: 2048 -> 4096), max_frame: 512
        * v004, num_mels: 128 -> 64, hop_size: 512 (window: 2048), max_frame: 512 -> 1024
- 12/19(土)
    - 今日やったこと
        * v003, num_mels: 128, hop_size: 512 -> 1024 (window: 2048 -> 4096), max_frame: 512(提出)
        * v004, num_mels: 128 -> 64, hop_size: 512 (window: 2048), max_frame: 512 -> 1024(提出)
        * CosineAnnealingLR適応
    - 次回やること
        * 後処理の分析EDA
- 12/22(火)
    - 今日やったこと
        * 後処理の分析EDA
            * 無音区間(ラベルなし区間)での予測がかなり間違えている
                * 無音もしくはノイズであることを明示的に伝えたい
        * noiseクラスを追加して学習n_class=25
    - 次回やること
        * EENDの論文を読む(Transformerの実装を確認して、わからない点を吉村さん林さんに確認する12/24まで)
        * center-loss実装
        * Time-stretchをしたときにwavデータのshapeに変化があるかどうかを確認
            * preprocessにて0.9, 1.1を追加する(ASRで実験的に良いAugmentationと言われている)
- 12/23(水)
    - 今日やったこと
        * center-loss実装
        * noiseクラスを追加して学習n_class=25(提出&記録)
        * noiseクラスのアノテーションを変更(ラベル区間を明示的に0に)
        * EENDの論文を読む(Transformerの実装を確認して、わからない点を吉村さん林さんに確認する12/24まで)
            * 60sec程度のかなり長い音を入力してアノテーションを付けることはできるか(無音やノイズの際に反応しないか)
    - 次回やること
        * EENDの論文を読む(Transformerの実装を確認して、わからない点を吉村さん林さんに確認する12/24まで)
        * Time-stretchをしたときにwavデータのshapeに変化があるかどうかを確認
            * preprocessにて0.9, 1.1を追加する(ASRで実験的に良いAugmentationと言われている)
- 12/24(木)
    - 今日やったこと
        * center-loss実装(提出)
        * ResNext50を実装してv000の実験
        * EENDの論文を読む(Transformerの実装を確認して、わからない点を吉村さん林さんに確認する12/24まで)
            * 60sec程度のかなり長い音を入力してアノテーションを付けることはできるか(無音やノイズの際に反応しないか)
    - 次回やること
        * EENDの論文を読む(Transformerの実装を確認して、わからない点を吉村さん林さんに確認する12/24まで)
        * Time-stretchをしたときにwavデータのshapeに変化があるかどうかを確認
            * preprocessにて0.9, 1.1を追加する(ASRで実験的に良いAugmentationと言われている)
- 12/25(金)
    - 今日やったこと
        * EENDの論文を読む(完全に理解した)
        * Time-stretchを実装v003-aug, v005で実験
            * preprocessにて0.9, 1.1を追加する(ASRで実験的に良いAugmentationと言われている)
    - 次回やること
        * pinknoiseをdataset内で変換できるようにwaveベースで実装&音を聴いて妥当性を評価
        * 実験結果を提出&まとめる
        * スケジューラーの無駄な減衰で学習が遅延しているので、BERTで使用されているやつをコピーして使用する
</div></details>
