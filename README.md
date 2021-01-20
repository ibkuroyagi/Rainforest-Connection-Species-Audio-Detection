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
- 発話区間予測 -> EEND-EDN実装
    * ダイアライゼーションの出力を学習データにして、inferenceも同じ分割にする(trainはランダムに分割するだけ)
- ~~ResNexst50をtorchvisionから重みをインポートしてファインチューニング(埋め込み層を明示的に作成)~~
- transformerでSED実装
- conformerでSED実装
- ~~Mixupを実装~~
- ~~Augmentationがデータセット内でできるようにwave形式の入力を受け取るようにdataset, collater_fcに追記~~
- ~~TimeStretchが0.9, 1.1のmatrix_tpを作成~~
- TTAでinferenceの汎か性能改善
- transformerで2048くらいのイメージをインプットにして動かす
    * 動かなければ1dconvで半分にダウンサンプリングする
- frameの予測は別のlayerで行い、そこはそこでBCEで最適化する.
    - その出力に対してsigmoidかけたものをマスク的扱いにしてクラスのframeの予測に反映させる
- shunmuraさんの手法と大きく差はないが、スコアに乖離がある。すべて同じ条件にして足並みをそろえる。
    - [https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/208830](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/208830)
    - PANNs SED architecture
    - BCE Loss
    - Random 10sec clip
    - Using only tp file
    - MixUp
    - The base model is EfficientNet
- もしかしたら、preprocessとのアノテーションのずれが生じている可能性があるため、on the flyで実験をするコードを作成
    - 10sec -> 48000x10 point -> 938frame x 128melで実験をする(1fold no aug)

## 決定事項
- 初手のCVの切り方はiterative-stratificationを用いる
- 推論に使うデータをいい感じにクロップするためにダイアライゼーションコードを作成する
- augmentationは[dcase2020-task4-1st](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Miyazaki_108.pdf)にならって,mixup, time-shiftを導入
    - Time-shiftはラベリングの難易度が高いため(現在12/29)保留中, comformer, EEND作成後に実装開始
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
- CenterLossの正則化を強くしすぎると過学習を起こしてしまい、悪化する -> noiseクラスを許す(random=True)と~~inferenceとの差がなくなるので効果的~~
    - ↑ノイズを学習させるとinferenceで劇的な悪化を起こす
- 学習がゆっくり進むようにmax_frame2048, batch_size128がほどよい
    - max_frameは小さいほうがCVとPLでスコアが高い->局所的な動きを学習しやすいからだと推測
- PANNs事前学習の重みはかなり効果的
- たぶん、ゴミラベルがまぎれている気がする(何度聞いても全く認識できない鳥の音声がある)
### 実験したモデルたち
- cnn/v007(aug)
- ~~res/v002~~
- res/v003(aug)
- ~~cnn/v005~~
- ~~cnn/v005-red~~
- cnn/v008(mel128hop1024,aug)
- ~~cnn/v009~~
- ~~cnn/v009-red~~
- ~~cnn/v010~~
- ~~cnn/v011~~
- ~~cnn/v012~~
- ~~cnn/v012-sum~~
- ~~cnn/v013~~
- ~~cnn/v013-sum~~
- ~~cnn/v002-clip065(check for sp0.9,1.1)~~
- ~~cnn/v002-ckup065(check for spなし(モデルの変更が悪さをしたのか))~~
- tra/v000
- con/v000
- tra/v000-sum -> sumの方がいい
- con/v000-sum -> meanの方がいい
- tra/v001(check for mixup) -> mixupは過学習対策にかなり効果があることを確認
- con/v001(check for mixup) -> mixupは過学習対策にかなり効果があることを確認
- tra/v002(check for n_class=24)
- con/v002(check for n_class=24)
- con/v003(check for n_class24, frame_mask)
- ~~eff/v000 -> efficientnet-b0 attention (BCE weal-label, wave, 10sec)~~
- ~~eff/v001 -> efficientnet-b0 attention (BCE weal-label, mel128hop1024, 10sec)~~~
- ~~eff/v002 -> efficientnet-b0 simple (BCE weal-label, wave, 10sec)~~
- ~~eff/v003 -> efficientnet-b0 simple (BCE weal-label, mel128hop1024, 10sec)~~
- ~~v004 -> efficientnet-b0 simple (BCE weal-label, wave, 10sec, lr=1.0e-3) compair by v000~~
- ~~v005 -> efficientnet-b0 simple (BCE weal-label, mel128hop1024, 10sec, lr=1.0e-3) compair by v000~~
- ~~v006(lr:0.005),v007(lr:0.01)efficientnet-b0 simple,wave -> 最適なlrにあたりをつけるために調査~~
- ~~v008 wave,sp0.9,1.1 collater_fcの分割方法を変更-> 1フレームあたりに含まれる正解音の幅をよりランダムに近くする(1/4)PL:0.701~~
- ~~v009 mel128hop1024,sp0.9,1.1 collater_fcの分割方法を変更-> 1フレームあたりに含まれる正解音の幅をよりランダムに近くする(1/4)PL:0.685~~
- ~~v010 mixup0.2+v008 -> PL:0.756で多少の回復.(旧モデル程度にスコア回復)~~
- ~~dropoutが推論時にも効いていることが発覚したため修正eff/v011(モデル埋め込みのmax部分も削除)->0.750~~
- ~~dropoutが推論時にも効いていることが発覚したため修正cnn/v002-clip065(モデル埋め込みのmax部分も削除)-> PL:0.705~~
- ~~eff/v012(分割方法を修正)->PL:0.753~~
- ~~eff/v013(on the fly) BCE ->中断~~
- ~~eff/v014(on the fly) FrameClipLoss ->中断~~
- ~~eff/v015(on the fly) BCE+MixUP ->中断~~
- ~~eff/v016(fix collater_fc as same as on the fly) BCE->中断~~
- ~~eff/v017(fix collater_fc as same as on the fly) FrameClipLoss->中断~~
- ~~eff/v018(fix collater_fc as same as on the fly) BCE+MixUP(v015とまったく同じ~~
- ~~eff/v019(preprocess fix)捨てた周波数に重要な情報が乗っている説->inferenceのデータもon the flyにする必要あり~~
- ~~eff/v020(att,shinmuraさんの手法をコピーmixupなし)~~ PL:0.791
- ~~eff/v021(att,shinmuraさんの手法をコピーmixup0.2)~~ PL:0.762
- ~~eff/v022(att,dializer_loss,mixupなし)~~ PL:0.797
- ~~eff/v023(att,dializer_loss,mixup0.1)~~ PL:0.818
- ~~eff/v024(att,mixup0.1,augmentation)~~:PL0.7530
- ~~eff/v025(att,dializer_loss,mixup0.1,raw)~~:PL0.801
- ~~eff/v026(att,mixupなし,torchでfeat作る)~~(CV:0.783495,PL:0.737,3000)
- eff/v026(att,mixupなし,torchでfeat作る)6000
- eff/v027(att,mixup0.2,torchでfeat作る)
- eff/v028(att,mixup0.2,augmentation,torchでfeat作る)
- ~~eff/v029(att,dializer_loss,mixup0.1,specaug)~~:PL:0.7696
- ~~eff/v030(att,dializer_loss,mixup0.1,center)~~CV:0.773921,PL:0.832
- tra/v003(mixupなし,dializer_loss)
- ~~cnn/v014(att,mixupなし,torchでfeat作る)~~CV:0.799761, PL:0.793
- cnn/v015(att,mixup0.2,augmentation,torchでfeat作る)
- ~~cnn/v016(att,dializer_loss,mixup0.1)~~CV:0.775324,PL:0.776
- ~~cnn/v017(att,dializer_loss,center,mixup0.1)~~CV:0.761139,PL:0.804
- ~~res/v004~~CV:0.775717,PL:0.723
- mob/v000
### 今の課題は何?
- v002の時代はCV,PLともに0.80のオーダーだったが、sp0.9,1,1に変更orモデルのクラスを25に変更にしたことでPLのスコアが下がった
- 原因を探るために、
    - v002-clip065(n_class=24, sp0.9,1.1)で実験->過学習対策にgood
    - v002-clip065(n_class=24, spなし)
- n_class24の方がPLが明らかに改善したので、n_class24で実験を進める

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
- 12/26(土)
    - 今日やったこと
        * CenterLossがモデル精度下げてることを発見したので、影響を下げる
        * Att部分でsumでaggregateしていた部分をmaxに変更->時間方向で最大値を採用することで長時間発生する音ではなく、actvivateしたという確率の部分により着目したモデルに修正
    - 次回やること
        * trainの分布をinferenceと同じにするために学習データの分割方法を中心にするだけでなく、完全にノイズのみのデータを許容するように修正
- 12/27(日)
    - 今日やったこと
        * trainの分布をinferenceと同じにするために学習データの分割方法を中心にするだけでなく、完全にノイズのみのデータを許容するように修正
        * ResNext50でも同様の処理を作成する
        * lrを小さくする、batch_sizeを大きくすることで学習曲線を緩やかにすることで過学習が生まれにくくする
    - 次回やること時間方向で最大値を採用することで長時間発生する音ではなく、actvivateしたという確率の部分により着目したモデルに修正
        * waveをデータセット内でfeatsに変換できるためのメソッドを作成
        * waveベースのAugmentationをDataset内で追加できるように
- 12/28(月)
    - 今日やったこと
        * waveをデータセット内でfeatsに変換できるためのメソッドを作成
        * waveベースのAugmentationをDataset内で追加できるように
    - 次回やること
        * 時間方向で最大値を採用することで長時間発生する音ではなく、actvivateしたという確率の部分により着目したモデルに修正
        * Mixupをpytorchの関数としてbatchを入力して(x*2, y*2) -> (x, y)となるように作成する
- 12/29(火)
    - 今日やったこと
        * mixup実装
        * randomの結果を確認->metricを25にしたせいで学習中を正しく評価できなくなった
            * 結果的にはかなり精度は悪化した
        * v005,v005-red(random=True) center-loss比較
        * v009,v009-red(random=False) center-loss比較
        * v009,v010(random=False) mixup比較
    - 次回やること
        * transformerを実装して動かす
        * v009, v010を提出してスコア記録
- 12/30(水)
    - 今日やったこと
        * mixupバグ修正
        * v005,v005-red(random=True) center-loss比較
        * v009,v009-red(random=False) center-loss比較
        * v009,v010(random=False) mixup比較
    - 次回やること
        * transformerを実装して動かす
        * v009, v010を提出してスコア記録
- 12/31(木)
    - 今日やったこと
        * v009, v010を提出してスコア記録
        * v011,v010 reduction:mean,sumで比較(frameの重要性を比較)
        * transformerを実装して動かす
            * 弱ラベルに対応するようにclsシンボルを時系列の先頭に追加する
        * cosformerを実装して動かす
            * 弱ラベルに対応するようにclsシンボルを時系列の先頭に追加する
        * res/v002を回収
    - 次回やること
        * v005,red,v009,red,v010提出
        * transformer関連のバグ取り
- 1/1(金)
    - 今日やったこと
        * v005,red,v009,red,v010提出
        * transformer関連のバグ取り
    - 次回やること
        * conformer関連バグとり
        * v012提出(frame512確認)
- 1/2(土)
    - 今日やったこと
        * conformer関連バグとり
        * v012提出(frame512確認)
    - 次回やること
        * v012での25を見てpost-processを工夫
        * その他モデルを提出
- 1/3(日)
    - 今日やったこと
        * v012での25を見てスコアが低くなってしまって原因を探る
        * その他モデルを提出
    - 次回やること
        * v012モデルで発話区間推定の活用を探る
        * transformer,conformerにSpecAugを実装
- 1/4(月)
    - 今日やったこと
        * v012での25を見てスコアが低くなってしまって原因を探る
        * transformer,conformerにSpecAugを実装
        * v002-clip065で対象実験をしてspかモデルかを判断
    - 次回やること
        * v012モデルで発話区間推定の活用を探る
- 1/5(火)
    - 今日やったこと
        * dialization-loss,clip-loss,frame-loss,center-lossのそれぞれを監視できるように修正
    - 次回やること
        * shinmura0さんのディスカッションの手法で実験
- 1/6(水)
    - 今日やったこと
        * efficientnet-b0, mobilenetv2を追加
        * shinmura0さんのディスカッションの手法で実験
            * v000 -> efficientnet-b0 attention (BCE weal-label, wave, 10sec)
            * v001 -> efficientnet-b0 attention (BCE weal-label, mel128hop1024, 10sec)
            * v002 -> efficientnet-b0 simple (BCE weal-label, wave, 10sec)
            * v003 -> efficientnet-b0 simple (BCE weal-label, mel128hop1024, 10sec)
            * v004 -> efficientnet-b0 simple (BCE weal-label, wave, 10sec, lr=1.0e-3) compair by v000
            * v005 -> efficientnet-b0 simple (BCE weal-label, mel128hop1024, 10sec, lr=1.0e-3) compair by v000
    - 次回やること
        * v012モデルで発話区間推定の活用を探る
- 1/7(木)
    - 今日やったこと
        * v000 -> efficientnet-b0 attention (BCE weal-label, wave, 10sec)
        * v001 -> efficientnet-b0 attention (BCE weal-label, mel128hop1024, 10sec)
        * v002 -> efficientnet-b0 simple (BCE weal-label, wave, 10sec)
        * v003 -> efficientnet-b0 simple (BCE weal-label, mel128hop1024, 10sec)
        * v004 -> efficientnet-b0 simple (BCE weal-label, wave, 10sec, lr=1.0e-3) compair by v000
        * v005 -> efficientnet-b0 simple (BCE weal-label, mel128hop1024, 10sec, lr=1.0e-3) compair by v000
        - 上記すべて回収したが、結果は悪い。->lrに原因があると考えv008,v007で実験(間違っていた)->best lr: 1e-3
        - collater_fcに問題があると考え、データの分割条件をより難しくなるように修正
    - 次回やること
        - v008,v009の結果を回収し考察
        - inferenceのTTAを実装完了する 
- 1/8(金)
    - 今日やったこと
        - v008,v009の結果を回収し考察
        - inferenceのTTAを実装完了する 
    - 次回やること
        - データセットの作成段階での小さなバグ(アノテーションがずれている)の可能性があるためon the flyを実装
- 1/9(土)
    - 今日やったこと
        - データセットの作成段階での小さなバグの可能性があるためon the flyを実装
        - v010~v015実験開始
    - 次回やること
        - 事前に変換済みの特徴量の方も修正
- 1/10(日)
    - 今日やったこと
        - 事前に変換済みの特徴量の方も修正
        - v016実験開始
    - 次回やること
        - eff/v013~v018, cnn/v002-clip065回収
        - もし、上記のスコアが悪い場合は、本質的に重藤な情報を捨ててしまっている可能性があるので、preprocessのパラメータを考察する
- 1/11(月)
    - 今日やったこと
        - eff/v013~v018, cnn/v002-clip065回収
        - もし、上記のスコアが悪い場合は、本質的に重藤な情報を捨ててしまっている可能性があるので、preprocessのパラメータを考察する
    - 次回やること
        - dializer_lossをattで適応
        - preprocess stage0実行
- 1/12(火)
    - 今日やったこと
        - dializer_lossをattで適応
        - preprocess stage0実行
        - on the flyをモデルでできるように実装
    - 次回やること
        - on the flyをモデルでできるように実装
        - 各種mixup,augなどの関連を調べる
- 1/13(水)
    - 今日やったこと
        - transformerでの実装見直し
        - cnnでdializer_lossについて確認
    - 次回やること
        - 実験結果を回収
        - 推論部を実装
        - スプレッドシートに実験を記録開始する
- 1/14(木)
    - 今日やったこと
        - v024回収
    - 次回やること
        - v025提出
- 1/15(木)
    - 今日やったこと
        - v025提出
    - 次回やること
        - v025提出
        - validのaccの割合をクラス単位で可視化(バランシングが効果ありそうなら実装)
        - 
</div></details>
