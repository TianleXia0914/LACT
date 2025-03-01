#!/usr/bin/env python 
# -*- coding: utf-8 -*-


import torch
import datasets
import transformers

import numpy as np

from itertools import chain
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

logger = datasets.logging.get_logger(__name__)
IGNORE_LABEL_ID = -100


class SFTDataSet(object):
    def __init__(self, args, tokenizer: transformers.tokenization_utils.PreTrainedTokenizerBase, model_type):
        data_files = {}
        logger.info(f"Args: {args}")
        if args.train_file:
            data_files["train"] = args.train_file
        if args.valid_file:
            data_files["validation"] = args.valid_file
        if args.test_file:
            for test_file in args.test_file:
                file_name = Path(test_file).stem
                data_files[f"test_{file_name}"] = test_file

        streaming = False
        if args.use_dialog_loss:
            streaming = True

        self.data_set = load_dataset("json", data_files=data_files, streaming=streaming)
        self.model_type = model_type

        self.padding_side = "left"

        if self.model_type == "QWen":
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.trunc_side = args.trunc_side

        self.instruction = args.instruction

        if args.use_dialog_loss:
            self.data_set = self.data_set.map(
                self.transform_for_dialog, batched=True, remove_columns=["messages"]
            )
        else:
            for split in self.data_set:
                self.data_set[split] = self.data_set[split].map(
                    self.transform, fn_kwargs={"is_inference": split.startswith("test_")}
                )


    def transform(self, example, is_inference=False):
        instance = self.convert_single_instance(example)
        if instance is None:
            feature = {
                "input_ids": np.random.randint(low=0, high=32000, size=self.max_length, dtype=np.int64),
                "attention_mask": [1 for _ in range(self.max_length)]
            }
            if not is_inference:
                feature["labels"] = [IGNORE_LABEL_ID] * self.max_length
        else:
            input_ids, labels, prompt_ids, answer_ids = instance
            if not is_inference:
                feature = {
                    "input_ids": input_ids,
                    "attention_mask": [1 if k != self.tokenizer.pad_token_id else 0 for k in input_ids],
                    "labels": labels
                }
            else:
                feature = {
                    "input_ids": prompt_ids,
                    "attention_mask": [1 for _ in prompt_ids],
                    # "labels": answer_ids
                }
        return feature

    def collate(self, batch, return_tensors=None):
        model_input = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }
        for instance in batch:
            model_input["input_ids"].append(instance["input_ids"])
            if "labels" in instance:
                model_input["labels"].append(instance["labels"])
            model_input["attention_mask"].append(instance["attention_mask"])

        if len(model_input["labels"]) == 0:
            model_input.pop("labels")

        model_input = self.pad(model_input)

        return model_input

    def convert_single_instance(self, instance):
        if "instruction" in instance:
            instruction, inputs, output = instance["instruction"], instance["input"], instance["output"]
            inputs = inputs.strip() + "\n"
        else:
            raise ValueError(f"No support format: {instance}")

        if self.instruction is not None:
            instruction = self.instruction

        prompt, answer = self._build_prompt(instruction, inputs, output)
        encode_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        encode_answer = self.tokenizer.encode(answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if self.model_type == "QWen":
            raw = self.tokenizer.encode(prompt + answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        else:
            raw = self.tokenizer.encode(prompt + " " + answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        try:
            assert raw == (encode_prompt + encode_answer)
        except AssertionError:
            logger.warning(f"Tokenized mismatch:\nRaw: {raw}\nNow: {encode_prompt + encode_answer}")
            return None

        input_ids = encode_prompt + encode_answer
        labels = [-100] * len(encode_prompt) + encode_answer

        truncated = self.trunc(input_ids, labels, encode_prompt, encode_answer)
        return truncated

    @staticmethod
    def _build_prompt(instruction, inputs, output):
        if instruction != "" and inputs != "":
            prompt = f"{instruction}\n{inputs}"
        elif inputs != "":
            prompt = inputs
        else:
            prompt = instruction
        answer = output
        return prompt, answer

    @staticmethod



    def trunc(self, input_ids, labels, encode_prompt, encode_answer):
        if len(input_ids) > self.max_length:
            to_trim = len(input_ids) - self.max_length

            if self.trunc_side == "left":
                if to_trim > len(encode_prompt):
                    logger.warning(
                        f"Truncating entire prompt. {self.tokenizer.decode(input_ids)}"
                    )
                    return None
                input_ids = input_ids[to_trim:]
                labels = labels[to_trim:]
                encode_prompt = encode_prompt[to_trim:]

            else:
                if to_trim > len(encode_answer):
                    logger.warning(
                        f"Truncating entire answer. {self.tokenizer.decode(input_ids)}"
                    )
                    return None
                input_ids = input_ids[:-to_trim]
                labels = labels[:-to_trim]
                encode_answer = labels[:-to_trim]

            logger.warning(
                f"Truncating source on left from {self.max_length + to_trim} to {self.max_length}, "
                f"result: {self.tokenizer.decode(input_ids)}"
            )

        return input_ids, labels, encode_prompt, encode_answer

    def pad(self, model_input):
        padded_input = {}
        for key, value in model_input.items():
            if key == "labels":
                pad_id = IGNORE_LABEL_ID
            elif key.startswith("attention"):
                pad_id = 0
            else:
                pad_id = self.tokenizer.pad_token_id

            if self.padding_side == "left":
                value_tensor = [torch.tensor(v[::-1]) for v in value]
                padded_input[key] = torch.fliplr(pad_sequence(value_tensor, batch_first=True, padding_value=pad_id))
            else:
                value_tensor = [torch.tensor(v) for v in value]
                padded_input[key] = pad_sequence(value_tensor, batch_first=True, padding_value=pad_id)
        return padded_input
