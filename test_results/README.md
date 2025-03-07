# 구분 기준 
* `./`: original
* `./test0`: s64(qkv-dim=64)
* `./test1`: s64 + decoder 측정 기준 변경 (beam search 구현 x)

---

# 테스트용 코드 
* latency dataset   
    * `--testn`의 숫자를 통해 데이터셋 저장 경로명을 지정

```sh
python latency_dataset_test.py \
--configs=configs/iwslt14.de-en/latency_dataset/npu.yml \
--testn=1 --latnpu

python latency_dataset_test.py \
--configs=configs/wmt14.en-de/latency_dataset/npu.yml \
--testn=1 --latnpu

python latency_dataset_test.py \
--configs=configs/wmt14.en-fr/latency_dataset/npu.yml \
--testn=1 --latnpu

python latency_dataset_test.py \
--configs=configs/wmt19.en-de/latency_dataset/npu.yml \
--testn=1 --latnpu
```

* latency predictor  
    * `--lat-dataset-path`로 학습에 사용할 데이터셋 지정

```sh
python latency_predictor.py \
--configs=configs/iwslt14.de-en/latency_predictor/npu.yml \
--lat-dataset-path=./latency_dataset/test1/iwslt14deen_npu.csv

python latency_predictor.py \
--configs=configs/wmt14.en-de/latency_predictor/npu.yml  \
--lat-dataset-path=./latency_dataset/test1/wmt14ende_npu.csv

python latency_predictor.py \
--configs=configs/wmt14.en-fr/latency_predictor/npu.yml \
--lat-dataset-path=./latency_dataset/test1/wmt14enfr_npu.csv

python latency_predictor.py \
--configs=configs/wmt19.en-de/latency_predictor/npu.yml \
--lat-dataset-path=./latency_dataset/test1/wmt19ende_npu.csv
```

* evo search  
    * 두 번째 명령어에 서치 과정 저장 경로명을 작성

```sh
python evo_search.py \
--configs=configs/iwslt14.de-en/supertransformer/space1.yml \
--evo-configs=configs/iwslt14.de-en/evo_search/iwslt14deen_npu.yml \
--cpu --num-workers=8 \
> ./test_results/test1/iwslt14.de-en.txt 2>&1

python evo_search.py \
--configs=configs/wmt14.en-de/supertransformer/space0.yml \
--evo-configs=configs/wmt14.en-de/evo_search/wmt14ende_npu.yml \
--cpu --num-workers=8 \
> ./test_results/test1/wmt14.en-de.txt 2>&1

python evo_search.py --configs=configs/wmt14.en-fr/supertransformer/space0.yml \
--evo-configs=configs/wmt14.en-fr/evo_search/wmt14enfr_npu.yml \
--cpu --num-workers=8 \
> ./test_results/test1/wmt14.en-fr.txt 2>&1

python evo_search.py \
--configs=configs/wmt19.en-de/supertransformer/space0.yml \
--evo-configs=configs/wmt19.en-de/evo_search/wmt19ende_npu.yml \
--cpu --num-workers=8 \
> ./test_results/test1/wmt19.en-de.txt 2>&1
```

---

# 기타

* beam search 부분을 어떻게 구현해야 할 지 모르겠음(인퍼런스 시에 어떤 방식으로 작동하는 지 모르겠음. 확인 필요)

