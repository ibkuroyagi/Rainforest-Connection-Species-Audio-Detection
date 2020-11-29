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

<details><summary>kaggle日記</summary><div>

- 11/29(日)
    - 今日やったこと
        * リポジトリ作成&コンペの理解
    - 次回やること
        * 手元環境でのEDAとstage1の作成

</div></details>
