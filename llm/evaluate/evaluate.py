#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import PeftModel
from tqdm.auto import tqdm

from llm.data.instruct_data_set import build_data_loader
from llm.data.sft_data import SFTDataSet
from torch.utils.data import DataLoader
from llm import MODELS, TOKENIZERS, CONFIG


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str, nargs="+")
    parser.add_argument("--task", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_suffix", type=str, default="")

    parser.add_argument("--prompt_template", type=str, default=None)

    parser.add_argument("--use_lora", type=bool, default=False)
    parser.add_argument("--lora_checkpoint_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--trunc_side", type=str, default="left")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--use_dialog_loss", type=bool, default=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    config = None
    if args.model_config:
        config = CONFIG[args.model_type].from_pretrained(
            pretrained_model_name_or_path=args.model_config
        )
        tokenizer = TOKENIZERS[args.model_type].from_pretrained(pretrained_model_name_or_path=args.model_config)
    else:
        tokenizer = TOKENIZERS[args.model_type].from_pretrained(pretrained_model_name_or_path=args.model_dir)
    model = MODELS[args.model_type].from_pretrained(
        pretrained_model_name_or_path=args.model_dir,
        config=config
    )
    model.half()

    logger.info(f"Load Model from {args.model_dir}")

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    #     model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        logger.info(f"Load LoRA from {args.lora_checkpoint_dir}")
        model = PeftModel.from_pretrained(model, args.lora_checkpoint_dir)

    sft_data = SFTDataSet(
        args=args,
        tokenizer=tokenizer,
        model_type="LLaMA"
    )
    for test_set in [split for split in sft_data.data_set if split.startswith("test_")]:
        data_loader = DataLoader(
            dataset=sft_data.data_set[test_set],
            batch_size=4,
            collate_fn=sft_data.collate
        )

        evaluate_steps = len(data_loader)

        logger.info(f"Load test file from {args.test_file}")

        model, data_loader = accelerator.prepare(model, data_loader)

        logger.info(f"***** Running evaluating for {test_set} *****")
        logger.info(f"    Batch size: {args.batch_size}")
        logger.info(f"    Evaluate step: {evaluate_steps}")

        progress_bar = tqdm(range(evaluate_steps), disable=not accelerator.is_local_main_process)

        model.eval()

        with open(os.path.join(args.output_dir, f"{test_set}_result.jsonl"), "w") as fout:
            for batch in data_loader:
                input_length = batch["input_ids"].shape[1]

                with torch.no_grad():
                    raw_model = accelerator.unwrap_model(model)
                    outputs = raw_model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
                predicts = outputs[:, input_length:].detach().cpu().numpy()
                for response in tokenizer.batch_decode(predicts, skip_special_tokens=True):
                    fout.write(json.dumps({"response": response}, ensure_ascii=False) + "\n")

                progress_bar.update(1)
        fout.close()

        accelerator.wait_for_everyone()



if __name__ == '__main__':
    main()
