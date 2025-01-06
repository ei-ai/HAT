# 에러 기록용 파일 
* 해결하거나 수정한 부분은 해당 파일(errors/README.md)에
* 수정이 필요하거나 기록이 필요한 부분은 해당 레포지토리(errors/)에 파일 생성해서 기록

# Contents
* [supertransformer 학습 단계](#supertransformer-학습-단계)  
* [supertransformer > onnx 변환 단계](#supertransformer--onnx-변환-단계)  
* [supertransformer > onnx > rknn 변환 단계](#supertransformer--onnx--rknn-변환-단계)
    
## supertransformer 학습 단계
<details>
<summary>supertransformer 학습 단계</summary>
<div markdown=1>

1. `AttributeError`: module 'numpy' has no attribute 'float'  
    * 배경: Numpy 버전 1.20 이상에서 발생  
    * 문제: `np.float`가 더 이상 지원되지 않음  
    * 해결방안: `float` 이나 `np.float64`로 변경 필요
    * 해결: `./replace_npfloat.py` 코드 사용해서 대체함(np.float64으로 대체)  
  
2. `./fairseq/modules/multihead_attention_super.py` 파일 오류   
    * 배경: PyTorch view 객체 in-place 연산(*=) 지원 안됨  
    * 문제: 해당 파일 line 198 `q *= self.scaling` 존재
    * 해결방안: `q = q * self.scaling`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함   
    `./fairseq/modules/multihead_attention.py` 파일 오류  
    * 배경: PyTorch view 객체 in-place 연산(*=) 지원 안됨  
    * 문제: 해당 파일 line 162 `q *= self.scaling` 존재
    * 해결방안: `q = q * self.scaling`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함  

3. `UserWarning`: This overload of add_ is deprecated 
    * 배경: PyTorch 최신 버전에서 `_add` 형식 바뀜(add_(Tensor other, *, Number alpha=1))  
    * 문제: `./fairseq/optim/adam.py` line 142: `exp_avg.mul_(beta1).add_(1 - beta1, grad)` 존재
    * 해결방안: `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함  

4. `UserWarning`: This overload of addcmul_ is deprecated 
    * 배경: PyTorch 최신 버전에서 `_addcmul` 형식 바뀜(addcmul_(Tensor tensor1, Tensor tensor2, *, Number value=1))  
    * 문제: `./fairseq/optim/adam.py` line 143: `exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)` 존재
    * 해결방안: `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)`로 변경 필요
    * 해결: 해당 파일 해당 라인 수정함  

5. `OSError`: [Errno 24] Too many open files (개인적인 문제)
    * 배경: `ulimit -n`으로 확인해 본 결과 최대 256개....
    * 문제: 컴퓨터 세팅 문제 
    * 해결방안: 파일 최대 오픈 개수 제한을 늘리거나, DataLoader num_workers 줄이기
    * 해결: 병렬처리를 건드리면 안될 것 같아서 일단은 `ulimit -n 4096
`, `launchctl limit maxfiles 4096 8192
`으로 컴퓨터 세팅을 바꿈  

</div>
</details>

## supertransformer > onnx 변환 단계
<details>
<summary>supertransformer > onnx 변환 단계 </summary>
<div markdown=1>

1. `AttributeError`: 'dict' object has no attribute 'eval'
    * 배경: 원본 레포에서 제공하는 모델이 가중치만 저장되어 있는 상태라고 합니다 
    * 문제: `convert_to_onnx.py`, line 41, in main: `model.eval()`
    * 해결방안: 모델을 fairseq의 task 써서 빌드하기  
    * 해결: 해당 파일 해당 부분 수정함(model = task.build_model(args))

2. `TypeError`: forward() missing 1 required positional argument: 'prev_output_tokens' 
    * 배경: 코드 문제 
    * 문제: `convert_to_onnx.py`, line 67, in main: export_to_onnx(model, src_vocab_size, tgt_vocab_size, dataset_name)
    * 해결방안: WrapperModel, prepare_for_onnx_export_() 사용하는 방식으로 수정
    * 해결: 됨 

3. `TracerWarning` 
    1. TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
    2. Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    3. .item() 못쓰게 해야함
    * [참고](https://jseobyun.tistory.com/578)
    * [netron](https://netron.app/)


</div>
</details>

## supertransformer > onnx > rknn 변환 단계

## 앞으로 수정해야 할 파일들들
### 2.1 Generate a latency dataset  
`latency_dataset.py`  
npu에서 돌아가게 수정  
### 2.2 Train a latency predictor  
`latency_predictor.py` 
### 2.3 Run evolutionary search with a latency constraint  
`evo_search.py`   
### 3. Train a Searched SubTransformer
`train.py`  
-> & 모델 변환 후 돌아가는 것 확인  