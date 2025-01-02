
# HAT

## Contents
* [수정한 부분](#수정한-부분)  
* [실행 및 세팅](#실행-및-세팅)  
  * [Data Preparation](#data-preparation)  
  * [Training](#training)  
  * [Testing](#testing)  



## 수정한 부분
<details>
<summary>파이썬 라이브러리 버전 충돌 문제</summary>
<div markdown=1>

1. `AttributeError`: module 'numpy' has no attribute 'float'  
    * 발생: supertransformer 학습 단계 
    * 배경: Numpy 버전 1.20 이상에서 발생  
    * 문제: `np.float`가 더 이상 지원되지 않음  
    * 해결방안: `float` 이나 `np.float64`로 변경 필요
    * 해결: `./replace_npfloat.py` 코드 사용해서 대체함(np.float64으로 대체)  
  
2. `./fairseq/modules/multihead_attention_super.py` 파일 오류  
    * 발생: supertransformer 학습 단계 
    * 배경: PyTorch view 객체 in-place 연산(*=) 지원 안됨  
    * 문제: 해당 파일 line 198 `q *= self.scaling` 존재
    * 해결방안: `q = q * self.scaling`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함 
    `./fairseq/modules/multihead_attention.py` 파일 오류  
    * 발생: supertransformer 학습 단계 
    * 배경: PyTorch view 객체 in-place 연산(*=) 지원 안됨  
    * 문제: 해당 파일 line 162 `q *= self.scaling` 존재
    * 해결방안: `q = q * self.scaling`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함  

3. `UserWarning`: This overload of add_ is deprecated 
    * 발생: supertransformer 학습 단계 
    * 배경: PyTorch 최신 버전에서 `_add` 형식 바뀜(add_(Tensor other, *, Number alpha=1))  
    * 문제: `./fairseq/optim/adam.py` line 142: `exp_avg.mul_(beta1).add_(1 - beta1, grad)` 존재
    * 해결방안: `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함  

4. `UserWarning`: This overload of addcmul_ is deprecated
    * 발생: supertransformer 학습 단계 
    * 배경: PyTorch 최신 버전에서 `_addcmul` 형식 바뀜(addcmul_(Tensor tensor1, Tensor tensor2, *, Number value=1))  
    * 문제: `./fairseq/optim/adam.py` line 143: `exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)` 존재
    * 해결방안: `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함  

5. `OSError`: [Errno 24] Too many open files (개인적인 문제)
    * 발생: supertransformer 학습 단계 
    * 배경: `ulimit -n`으로 확인해 본 결과 최대 256개....
    * 문제: 컴퓨터 세팅 문제 
    * 해결방안: 파일 최대 오픈 개수 제한을 늘리거나, DataLoader num_workers 줄이기
    * 해결: 병렬처리를 건드리면 안될 것 같아서 일단은 `ulimit -n 4096
`, `launchctl limit maxfiles 4096 8192
`으로 컴퓨터 세팅을 바꿈  

6. `AttributeError`: 'dict' object has no attribute 'eval'
    * 발생: supertransformer > onnx 변환 단계  
    * 배경: 원본 레포에서 제공하는 모델이 가중치만 저장되어 있는 상태라고 합니다 
    * 문제: `convert_to_onnx.py`, line 41, in main: `model.eval()`
    * 해결방안: 모델을 fairseq의 task 써서 빌드하기  
    * 해결: 해당 파일 해당 부분 수정함(model = task.build_model(args))

7. `TypeError`: forward() missing 1 required positional argument: 'prev_output_tokens'
    * 발생: supertransformer > onnx 변환 단계  
    * 배경: 
    * 문제: `convert_to_onnx.py`, line 67, in main: export_to_onnx(model, src_vocab_size, tgt_vocab_size, dataset_name)
    * 해결방안:  
    * 해결: 수정중....

8. `TracerWarning` 
    1. TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
    2. Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    3. .item() 못쓰게 해야함
    * [참고](https://velog.io/@dust_potato/ONNX%EB%9E%80-1-%EA%B8%B0%EB%B3%B8-%EC%84%A4%EB%AA%85%EA%B3%BC-%EC%9B%90%EB%A6%AC)


</div>
</details>



## 실행 및 세팅
### Dependencies
<details>
<summary> [MINSEO KIM](https://github.com/440g) </summary>
<div markdown=1>

* OS: Sonoma 14.1
* GPU: 14개(Apple M3 Pro)
* Python = 3.9.21
* requirements.txt
    ```sh
    build==1.2.2.post1
    cffi==1.17.1
    click==8.1.8
    colorama==0.4.6
    coloredlogs==15.0.1
    ConfigArgParse==1.7
    Cython==3.0.11
    -e git+https://github.com/mit-han-lab/hardware-aware-transformers.git@70e5a279d080670208249fdd98ed731fa9bcc466#egg=fairseq
    fastBPE @ file:///Users/(생략)/hardware-aware-transformers/fastBPE
    filelock==3.16.1
    flatbuffers==24.12.23
    fsspec==2024.12.0
    humanfriendly==10.0
    importlib_metadata==8.5.0
    Jinja2==3.1.5
    joblib==1.4.2
    lxml==5.3.0
    MarkupSafe==3.0.2
    mpmath==1.3.0
    networkx==3.2.1
    numpy==2.0.2
    onnx==1.17.0
    onnxruntime==1.19.2
    packaging==24.2
    portalocker==3.0.0
    protobuf==5.29.2
    pycparser==2.22
    pyproject_hooks==1.2.0
    regex==2024.11.6
    sacrebleu==2.4.3
    sacremoses==0.1.1
    sympy==1.13.1
    tabulate==0.9.0
    tensorboardX==2.6.2.2
    tomli==2.2.1
    torch==2.5.1
    tqdm==4.67.1
    typing_extensions==4.12.2
    ujson==5.10.0
    zipp==3.21.0
    ```

</div>
</details>



### Data Preparation
```sh
bash configs/[task_name]/get_preprocessed.sh
```
```sh
bash configs/wmt4.en-de/get_preprocessed.sh
bash configs/wmt14.en-fr/get_preprocessed.sh
bash configs/wmt19.en-de/get_preprocessed.sh
bash configs/iwslt14.de-en/get_preprocessed.sh
```


### Training
1. Train a supertransformer
    * train
    ```sh
    python train.py --configs=configs/wmt14.en-de/supertransformer/space0.yml
    python train.py --configs=configs/wmt14.en-fr/supertransformer/space0.yml
    python train.py --configs=configs/wmt19.en-de/supertransformer/space0.yml
    python train.py --configs=configs/iwslt14.de-en/supertransformer/space1.yml
    ```
    * download
    ```sh
    python download_model.py --model-name=[model_name]
    ```
    ```sh
    python download_model.py --model-name=HAT_wmt14ende_super_space0
    python download_model.py --model-name=HAT_wmt14enfr_super_space0
    python download_model.py --model-name=HAT_wmt19ende_super_space0
    python download_model.py --model-name=HAT_iwslt14deen_super_space1
    ```
    * convert(수정중)
    ```sh
    convert supertransformer
    python convert_onnx.py --dataset-name=[dataset_name] --configs=configs/[task_name]/convert_onnx/[search_space].yml
    ```
    ```sh
    python convert_onnx.py --dataset-name=wmt14ende --configs=configs/wmt14.en-de/convert_onnx/space0.yml
    python convert_onnx.py --dataset-name=wmt14enfr --configs=configs/wmt14.en-fr/convert_onnx/space0.yml
    python convert_onnx.py --dataset-name=wmt19ende --configs=configs/wmt19.en-de/convert_onnx/space0.yml
    python convert_onnx.py --dataset-name=iwslt14deen --configs=configs/iwslt14.de-en/convert_onnx/space1.yml
    ```

2. Evolutionary Search  
    2.1 Generate a latency dataset  
    2.2 Train a latency predictor  
    2.3 Run evolutionary search with a latency constraint  

3. Train a Searched SubTransformer



### Testing

