import os
import torch
import onnx
from onnxsim import simplify
from fairseq import tasks, options, utils
from wrapper_models import wrapper_onnx

def simplify_onnx(onnx_file_path):
    model = onnx.load(onnx_file_path)
    simplified_model, check = simplify(model)
    if check:
        onnx_file_path = onnx_file_path[0:-5] + "_sim.onnx"
        onnx.save(simplified_model, onnx_file_path)
        print("| ONNX model simplified")
    else:
        print("| ONNX model could not simplified")

def generate_dummy_data(args):
    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 25, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([dummy_sentence_length])
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)

    return src_tokens_test, src_lengths_test, prev_output_tokens_test_with_beam


def export_to_onnx(model, args):
    if args.train_subtransformer:
        onnx_file_path = f"./onnx_models/{args.sub_model_name}"
    else:
        onnx_file_path = f"./onnx_models/{args.data.removeprefix('data/binary/')}"

    if args.enc:
        onnx_file_path += "_enc.onnx"
    elif args.dec:
        onnx_file_path += "_dec.onnx"
    else:
        onnx_file_path += ".onnx"

    os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

    src_tokens, src_lengths, prev_output_tokens = generate_dummy_data(args)
    inputs, input_names, output_names = None, None, None

    if args.enc:
        model = model.encoder
        print(f"| Encoder Arch: {model} \n")
        inputs = (src_tokens, src_lengths)
        input_names = ["src_tokens", "src_lengths"]
        output_names = ["encoder_output"]

    elif args.dec:
        encoder_out_test = model.encoder(src_tokens, src_lengths) if model.encoder else {"encoder_padding_mask": None}
        
        bsz = src_tokens.size(0)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
        encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)
        incre_states = {}
        
        model = model.decoder
        print(f"| Decoder Arch: {model} \n")
        inputs = (prev_output_tokens, encoder_out_test_with_beam, incre_states)
        input_names = ["prev_output_tokens", "encoder_out", "incre_states"]
        output_names = ["decoder_output"]

    else:
        inputs = (src_tokens, src_lengths, prev_output_tokens)
        input_names = ["src_tokens", "src_lengths", "prev_output_tokens"]
        output_names = ["model_output"]

    model = wrapper_onnx.WrapperModelONNX(model)
    model.prepare_for_onnx_export()

    torch.onnx.export(
        model,
        inputs,
        onnx_file_path,
        opset_version=14,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None
    )

    print(f"| Successfully saved ONNX model at: {onnx_file_path}")

    simplify_onnx(onnx_file_path)


def main():
    parser = options.get_converting_parser()
    args = options.parse_args_and_arch(parser)
    print(f"| Configs: {args}")

    print(f"| Buildng model {args.arch}...")
    task = tasks.setup_task(args)
    model = task.build_model(args)
    with torch.no_grad():
        config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)
        model.set_sample_config(config_sam)
    model.eval()


    if args.train_subtransformer:
        print(" \n\n| Exporting SubTransformer model to ONNX...\n")
        print(f"| SubTransformer Arch: {utils.get_subtransformer_config(args)} \n")
    else:
        print(" \n\n| Exporting SuperTransformer model to ONNX...\n")
        print(f"| SuperTransformer Arch: {model} \n")
    if args.enc:
        print(" | Encoder only\n")
    if args.dec:
        print(" | Decoder only\n")

    export_to_onnx(model, args)
    
    print("| All set!")


if __name__ == "__main__":
    main()