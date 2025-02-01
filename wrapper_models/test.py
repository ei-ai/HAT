import torch
import numpy as np

dummy_sentence_length = 30
dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
src_lengths_test = torch.tensor([30])
prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * 5, dtype=torch.long)
dummy_encoder_out_length = 512
encoder_out_test_with_beam = [[7] * dummy_encoder_out_length for _ in range(5)]
encoder_out_test_with_beam = torch.tensor([encoder_out_test_with_beam] * dummy_sentence_length, dtype=torch.long)

print("src_tokens_test",  src_tokens_test.shape)
src_tokens = src_tokens_test.numpy()
print("src_tokens",  src_tokens.shape)
src_tokens = src_tokens.reshape(1, 1, *src_tokens.shape)
print("src_tokens",  src_tokens.shape)

print("prev_output_tokens_test_with_beam",  prev_output_tokens_test_with_beam.shape)
prev_output_tokens = prev_output_tokens_test_with_beam.numpy()
print("prev_output_tokens",  prev_output_tokens.shape)
print("prev_output_tokens",  prev_output_tokens.shape)

print("encoder_out_test_with_beam",  encoder_out_test_with_beam.shape)
encoder_out = encoder_out_test_with_beam.numpy()
print("encoder_out",  encoder_out.shape)
encoder_out = encoder_out.reshape(1, *encoder_out.shape)
print("encoder_out",  encoder_out.shape)