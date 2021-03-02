# コンペの解法まとめ
## shinmura0さん
### このコンペの肝
- missing-label
- ラベル長がずれている
    - ↑に対して、ハンドラベルすると解決
### 前処理
- スペクトログラムを850x850にしたものがbest(大きいほど良い)
    - 0.69 -> 0.81
### 学習
- MixUpを確率的0.5に削るといい感じに
### 後処理
- 0.01secは認識できないので、超短時間のイベントを消すために、時間方向の移動平均をとる(SED特有で有向)
    - 0.078 -> 0.840
### 上位陣の解法
missing label対策
- mask loss(正確なラベルだけを学習)
- mask周波数(がっつり上の部分を消す)

## yukiさん
- 256x1001のメルスぺ(60/9=6.6sec)
- Adam -> SGDにして学習安定
- lr: 0.15 (かなり高い)
adversarial validation
AUC:0.81 -> 収音環境、収音機材の違い
train寄りのtrainを除くと精度微増
- データがどう作られたかを良く調べなければならなかった

## araiさん
[https://speakerdeck.com/koukyo1994/niao-wa-konpefan-sheng-hui-zi-liao](https://speakerdeck.com/koukyo1994/niao-wa-konpefan-sheng-hui-zi-liao)
positive and unlabeled learning
事前確率シフト
- 画像分類アプローチ
    - 出てきた周波数のみを用いて計算
    - 重複する周波数をゆるして確率出力
    - SAM, cossineannearing T_max=10
    - lossのマスクや、入力のマスクによって
    - resizeはこれ
```
transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
````
- SEDアプローチ
    - 0.82=0.88スコアをアンサンブルして0.902くらいまでレベルアップ

## きょうへいさん
[https://speakerdeck.com/kuto5046/niao-wa-konpefan-sheng-hui-birdcall-revengetimu](https://speakerdeck.com/kuto5046/niao-wa-konpefan-sheng-hui-birdcall-revengetimu)
ラベルの分布の違い
### stage1
割と平凡なモデル
CV:0.81/PL:0.84  
SED  
TPのみを使用  
30epoch  
LSEPLoss (FAT20019 3rd)  
augmenationなし  

### stage2
stage1を強化してpseude-labelを作成  
CV: 0.734/PL0.896  
- TP and FPを用いてFocal Loss ベースのmask loss
    - ラベルを3つに分解
        - 1:TPラベル(正例)
        - 0:曖昧なラベル
        - -1:FPラベル(負例)
ただし、クラス数を増加させているだけ(実際は二値分類をしている。その中で、BCEの一部の損失を更新させないようにする)
stage1では学習効率悪いが、stage2でファインチューニングで少ないepochでスコア改善
5epochだけ

### stage3
stage2でのpseudo-labelをしたものを適応
5epochだけ


### 今後結果を残すために
今抱えている課題をすべて書き出す  
その後、最小単位ごとに流行りの解決方法を列挙し模索する  
特に過去解法はかなり参考になる情報があるので、それらを抜け漏れなく収集する必要がある  
検索ワードのセンス
- [https://speakerdeck.com/koukyo1994/tuo-deepdepon-haiparatiyuninguyun-ren-wozu-ye-surutameni](https://speakerdeck.com/koukyo1994/tuo-deepdepon-haiparatiyuninguyun-ren-wozu-ye-surutameni)
