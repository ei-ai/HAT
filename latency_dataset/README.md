* `./`: original
* `./test0`: s64(qkv-dim=64)
* `./test1`: s64 + decoder 측정 기준 변경 (beam search 구현 x)
---
* beam search 부분을 어떻게 구현해야 할 지 모르겠음(인퍼런스 시에 어떤 방식으로 작동하는 지 모르겠음. 확인 필요)
* test0 제외한 데이터셋은 아직 제대로 안 올림

```sh
python latency_dataset_test.py --latnpu --configs=configs/iwslt14.de-en/latency_dataset/npu.yml
python latency_dataset_test.py --latnpu --configs=configs/wmt14.en-de/latency_dataset/npu.yml
python latency_dataset_test.py --latnpu --configs=configs/wmt14.en-fr/latency_dataset/npu.yml
python latency_dataset_test.py --latnpu --configs=configs/wmt19.en-de/latency_dataset/npu.yml
```

```sh
python latency_predictor.py --configs=configs/iwslt14.de-en/latency_predictor/npu.yml --lat-dataset-path=./latency_dataset/test1/iwslt14deen_npu.csv
python latency_predictor.py --configs=configs/wmt14.en-de/latency_predictor/npu.yml --lat-dataset-path=./latency_dataset/test1/wmt14ende_npu.csv
python latency_predictor.py --configs=configs/wmt14.en-fr/latency_predictor/npu.yml --lat-dataset-path=./latency_dataset/test1/wmt14enfr_npu.csv
python latency_predictor.py --configs=configs/wmt19.en-de/latency_predictor/npu.yml --lat-dataset-path=./latency_dataset/test1/wmt19ende_npu.csv
```

```sh
python evo_search.py --configs=configs/wmt14.en-de/supertransformer/space0.yml --evo-configs=configs/wmt14.en-de/evo_search/wmt14ende_npu.yml --cpu --num-workers=8
python evo_search.py --configs=configs/wmt14.en-fr/supertransformer/space0.yml --evo-configs=configs/wmt14.en-fr/evo_search/wmt14enfr_npu.yml --cpu --num-workers=8
python evo_search.py --configs=configs/wmt19.en-de/supertransformer/space0.yml --evo-configs=configs/wmt19.en-de/evo_search/wmt19ende_npu.yml --cpu --num-workers=8
python evo_search.py --configs=configs/iwslt14.de-en/supertransformer/space1.yml --evo-configs=configs/iwslt14.de-en/evo_search/iwslt14deen_npu.yml --cpu --num-workers=8
```
