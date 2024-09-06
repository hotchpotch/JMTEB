---
name: Leaderboard submission PR
about: 自作モデルの評価結果をリーダーボードに反映させるPR
title: "[Submission] "
labels: "submission"
assignees: ''
---

<!-- 
PRを出していただき、ありがとうございます。
base branchを`dev`にするよう、お願いいたします。
-->

## モデルの基本情報
**name**: 
**type**: <!-- バックボーンモデル，例えば BERT, LLaMA... -->
**size**:
**lang**: ja / multilingual

## モデル詳細
<!-- 
学習手法，学習データなど，モデルの詳細について記載してください
-->


## seen/unseen申告
JMTEBの評価データセットの中，training splitをモデル学習に使用した，またはvalidation setとして，ハイパラチューニングやearly stoppingに使用したデータセット名をチェックしてください。
* Classification
  * [ ] Amazon Review Classification
  * [ ] Amazon Counterfactual Classification
  * [ ] Massive Intent Classification
  * [ ] Massive Scenario Classification
* Clustering
  * [ ] Livedoor News
  * [ ] MewsC-16-ja
* STS
  * [ ] JSTS
  * [ ] JSICK
* Pair Classification
  * [ ] PAWS-X-ja
* Retrieval
  * [ ] JAQKET
  * [ ] Mr.TyDi-ja
  * [ ] JaGovFaqs-22k
  * [ ] NLP Journal title-abs
  * [ ] NLP Journal title-intro
  * [ ] NLP Journal abs-intro
* Reranking
  * [ ] Esci
* [ ] 申告しません


## 評価スクリプト
<!-- 
可能であれば評価用のスクリプトを記入してください。
モデルに合わせた特殊なセッティングは必ず書いてください。
-->

## その他の情報

## 動作確認
- [ ] テストが通ることを確認した
- [ ] マージ先がdevブランチであることを確認した
- [ ] 結果の`json`ファイルを正しい位置にアップロードした
- [ ] `leaderboard.md`を更新した
- [ ] ...

<!-- 
## その他
-->