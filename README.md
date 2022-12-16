# AI-Ground

## Project
  * 대회 : 2022 유플러스 AI Ground 대회 참가
  * 목표
      1. 추천시스템의 기초, 개념을 명확히 이해
      2. 추천시스템의 모델을 학습하기 위한 데이터의 형태 및 구조 확인
      3. 대회를 통해서 Transformer을 활용한 추천시스템 모델을 구현해보고 성능을 확인하기
      
## Dataset
  * LG 유플러스의 아이들나라 서비스 데이터
      1. 각 사람별로 언제 어느 영상을 봤는지에 대한 데이터
          1. Sequential Recommendation에 해당
      2. 각 영상 및  별로 특징이 존재
          1. 영상 : 키워드, 장르, 제작 나라
          2. 사람 : 성별, 나이 등등
  
## Baseline
  1. BERT4REC(https://arxiv.org/abs/1904.06690) 논문을 참고해서 베이스라인을 개발
      * 최대한 논문에 입각해서 BERT4REC 모델을 구현하고 이에 대한 성능을 확인해보기
      * 해당 모델이 가지는 추천 시스템에 대한 성능 및 특징 확인하기
  2. Huggingface의 transformers 라이브러리에 있는 Bert Model 코드를 활용
  3. 2번 모델에 사용자의 특징 및 각 영상의 키워드 데이터를 사용할 수 있게 모델을 고도화
 
## BERT4REC
  1. 대회를 진행하면서 파악한 BERT4REC의 중요 특징
      * End-to-End
          * 학습을 할 때는 Sequence 길의 특정 비율에 해당되는 만큼 중간 중간에 MASK 토큰을 씌우고 MASK에 해당되는 영상이 무엇인지를 학습한다.
          * 학습한 모델을 그래도 추론에 활용하는데 추론을 할 때는 Sequence 뒤에 MASK 토큰을 이어 붙이고 해당 토큰이 무슨 영상을 가리키는지 확인한다.
      * Model Structure
          * 모델 깊이 및 크기가 어느정도 커야지 MLM이 학습이 된다.
          * 본인은 BERT의 BASE hyperparameter를 그대로 사용하였다.
      * Training
          * Mask Probability가 너무 낮으면 모델 성능이 낮아진다.
          * 본인이 실험할 때는 0.4 ~ 0.6이 좋은 성능을 보여주었다.
      * Shared Embedding
          * Input Embedding을 그대로 Output Classification을 사용할 때 많은 성능 향상을 가져왔다.
          * Output Classifiaction Layer를 별도로 선언하면 과적합 
 
## Hyperparameter
|Hyperparameter|Value|
|--------|-----------|
|epochs|100|
|learning rate|1.2e-4|
|sequence max length|128|
|mask probability|0.6|
|hidden size|768|
|intermediate size|3072|
|num layers|12|
|dropout probabiility|0.1|
|num head|8|
|train batch size|128|
|weight decay|1e-3|
|warmup ratio|0.1|

## Terminal Command Example
  ```shell
  # Training (Signle Model)
  python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 1 \
    --max_length 128 \
    --keyword_max_length 20 \
    --learning_rate 1.2e-4 \
    --weight_decay 1e-2 \
    --epochs 100 \
    --train_batch_size 280 \
    --mlm_probability 0.6 \
    --logging_steps 100 \
    --save_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 4 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps/seed1
  ```

## Ensemble
  * 위 hyperparameter의 모델을 seed를 다르게 해서 5개의 모델을 학습하고 Soft Voting을 실행

## Leaderboard
|Model|Public|Private|
|-----|----|----|
|ensemable|0.2411(5th)|0.1719(24th)|

## 후기
  * 아쉬운 점
      1. Public 점수에 비해서 Private 점수가 많이 하락하였다.
      2. 문의한 결과 Public 점수의 기준은 5 & 6월, Private 점수의 기준은 6 & 7월 이였다.
      3. 즉 내가 학습하고 개발한 모델은 짧은 관점에서만 잘 추천을 해주고 장기적인 관점에서는 추천을 잘 해주지 못했다.
      4. 데이터에 대한 분석 및 전처리가 더 필요했을 것으로 판단. 
      5. Public Score가 잘 나오는 것을 확인하여 안심하고 다양한 모델을 앙상블 하지 않고 seed 앙상블을 한 것도 문제점이였다.
  * 배운 점 
      1. BERT4REC 모델이 잘 활용하면 추천에서 잘 사용될 수 있다는 점을 확인하였다.
      2. Sequential Recommendation에서 장기적으로 추천을 할 수 있는 모델이 무엇이 있는지에 대한 호기심이 들었다.
      3. 대회를 통해서 추천시스템의 기초 및 개념에 대한 내용을 짧은 시간안에 집중하여 공부할 수 있었다.
      









