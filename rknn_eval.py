import os
import subprocess
import time
import numpy as np
from rknn.api import RKNN

# 참고 utils
# def measure_latency(args, model, dummy_src_tokens, dummy_prev):
#     # latency measurement
#     assert not (args.latcpu and args.latgpu)

#     model_test = copy.copy(model)
#     model_test.set_sample_config(get_subtransformer_config(args))
#     src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
#     src_lengths_test = torch.tensor([30])
#     prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)

#     if args.latcpu:
#         model_test.cpu()
#         print('| Measuring model latency on CPU...')
#     elif args.latgpu:
#         # model_test.cuda()
#         src_tokens_test = src_tokens_test.cuda()
#         src_lengths_test = src_lengths_test.cuda()
#         prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()
#         src_tokens_test.get_device()
#         print('| Measuring model latency on GPU...')
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)

#     # dry runs
#     for _ in range(5):
#         encoder_out_test = model_test.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

#     encoder_latencies = []
#     print('| Measuring encoder...')
#     for _ in tqdm(range(args.latiter)):
#         if args.latgpu:
#             start.record()
#         elif args.latcpu:
#             start = time.time()

#         model_test.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

#         if args.latgpu:
#             end.record()
#             torch.cuda.synchronize()
#             encoder_latencies.append(start.elapsed_time(end))
#             if not args.latsilent:
#                 print('| Encoder one run on GPU: ', start.elapsed_time(end))

#         elif args.latcpu:
#             end = time.time()
#             encoder_latencies.append((end - start) * 1000)
#             if not args.latsilent:
#                 print('| Encoder one run on CPU: ', (end - start) * 1000)

#     # only use the 10% to 90% latencies to avoid outliers
#     print(f'| Encoder latencies: {encoder_latencies}')
#     encoder_latencies.sort()
#     encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]
#     print(f'| Encoder latency: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')

#     # beam to the batch dimension
#     # encoder_out_test_with_beam = encoder_out_test.repeat(1, args.beam)
#     bsz = 1
#     new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
#     if args.latgpu:
#         new_order = new_order.cuda()

#     encoder_out_test_with_beam = model_test.encoder.reorder_encoder_out(encoder_out_test, new_order)

#     # dry runs
#     for _ in range(5):
#         model_test.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
#                            encoder_out=encoder_out_test_with_beam)

#     # decoder is more complicated because we need to deal with incremental states and auto regressive things
#     decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
#     if 'iwslt' in args.arch:
#         decoder_iterations = decoder_iterations_dict['iwslt']
#     elif 'wmt' in args.arch:
#         decoder_iterations = decoder_iterations_dict['wmt']

#     decoder_latencies = []
#     print('| Measuring decoder...')
#     for _ in tqdm(range(args.latiter)):
#         if args.latgpu:
#             start.record()
#         elif args.latcpu:
#             start = time.time()
#         incre_states = {}
#         for k_regressive in range(decoder_iterations):
#             model_test.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
#                                encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
#         if args.latgpu:
#             end.record()
#             torch.cuda.synchronize()
#             decoder_latencies.append(start.elapsed_time(end))
#             if not args.latsilent:
#                 print('| Decoder one run on GPU: ', start.elapsed_time(end))

#         elif args.latcpu:
#             end = time.time()
#             decoder_latencies.append((end - start) * 1000)
#             if not args.latsilent:
#                 print('| Decoder one run on CPU: ', (end - start) * 1000)

#     # only use the 10% to 90% latencies to avoid outliers
#     decoder_latencies.sort()
#     decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]

#     print(f'| Decoder latencies: {decoder_latencies}')
#     print(f'| Decoder latency: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms\n')

#     print(f"| Overall Latency: {np.mean(encoder_latencies) + np.mean(decoder_latencies)}")



# 인코더 디코더 쪼개서 변환하는 것이 필요해 보인다. 일단은 임시방편으로 이렇게 설정 
def measure_latency_rknn_simulated(args, model_path, dummy_src_tokens, dummy_prev):
    rknn = RKNN()
    rknn.load_rknn(model_path)
    rknn.init_runtime(target='rk3588')

    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([30])
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)

    encoder_latencies = []
    decoder_latencies = []

    for _ in range(5):
        rknn.inference(inputs=[encoder_input])  
        rknn.inference(inputs=[decoder_input])  

    print("| Measuring Simulated Encoder Latency...")
    for _ in range(iterations):
        start_time = time.time()
        rknn.inference(inputs=[encoder_input])  
        end_time = time.time()
        encoder_latencies.append((end_time - start_time) * 1000)
    print(f'| Encoder latencies: {encoder_latencies}')
    encoder_latencies.sort()
    encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]
    print(f'| Encoder latency: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')

    # decoder is more complicated because we need to deal with incremental states and auto regressive things
    decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        decoder_iterations = decoder_iterations_dict['iwslt']
    elif 'wmt' in args.arch:
        decoder_iterations = decoder_iterations_dict['wmt']

    print("| Measuring Simulated Decoder Latency...")
    for _ in range(decoder_iterations):
        start_time = time.time()
        rknn.inference(inputs=[decoder_input])  
        end_time = time.time()
        decoder_latencies.append((end_time - start_time) * 1000)

    decoder_latencies.sort()
    decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -int(args.latiter * 0.1)]
    print(f'| Decoder latencies: {decoder_latencies}')
    print(f'| Decoder latency: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms\n')
    print(f"| Overall Latency: {np.mean(encoder_latencies) + np.mean(decoder_latencies)}")


    rknn.release()
    return 



def main(args):
    model_path = f'rknn_models/{args.model_name}/{args.model_name}.rknn'

    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError
    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    # # Measure model latency, the program will exit after profiling latency
    # if args.latcpu or args.latgpu:
    #     utils.measure_latency(args, model, dummy_src_tokens, dummy_prev)
    #     exit(0)
    measure_latency_rknn_simulated(args, model_path, dummy_src_tokens, dummy_prev)


if __name__ == "__main__":
    import sys # 나중에 다른 파일들처럼 형식 수정 
    if len(sys.argv) != 3:
        print("Usage: python rknn_eval.py <model_name> <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    input_data_path = sys.argv[2]
    main(dataset_name, input_data_path)
    main(args)
