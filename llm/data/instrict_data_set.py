#!/usr/bin/env python 
# -*- coding: utf-8 -*-


import copy
import json
import os

from datasets import load_dataset
from torch.utils.data import DataLoader


def build_data_loader(logger, accelerator: Accelerator, data_file, tokenizer, batch_size=8, max_length=1024,
                      stream=False, inference=False, prompt_template=None, model_type="chatGLM"):
    data_set = load_dataset("json", data_files=data_file, split="train", streaming=stream)
    prompt_fields = list(data_set.features.keys())

    def build_prompt(examples):
        if prompt_template is None:
            prompt = examples[prompt_fields[0]]
        else:
            if os.path.isfile(prompt_template):
                prompt = "".join(open(prompt_template, "r").readlines())
            else:
                prompt = prompt_template
            for field in prompt_fields:
                if ("[%s]" % field.upper()) in prompt:
                    prompt = prompt.replace("[%s]" % field.upper(), examples[field])
        examples["final_prompt"] = prompt

        return examples

    def transform(examples):
        if model_type == "chatGLM":
            encode = tokenizer(
                examples["final_prompt"], examples[prompt_fields[-1]], truncation=True, max_length=max_length
            )

            prompt_length = encode["input_ids"].index(tokenizer.bos_token_id)
            label_ids = [-100] * prompt_length + encode["input_ids"][prompt_length:]

            feature = {
                "input_ids": encode["input_ids"],
                "position_ids": encode["position_ids"],
                "attention_mask": encode["attention_mask"],
                "labels": label_ids
            }

        elif model_type == "LLaMA" or model_type == "Baichuan":
            prompt_ids = tokenizer.encode(examples["final_prompt"])
            encode = tokenizer(examples["final_prompt"] + examples[prompt_fields[-1]])

            label_ids = copy.deepcopy(encode["input_ids"])
            label_ids = [-100] * (len(prompt_ids)-1) + label_ids[len(prompt_ids)-1:]

            feature = {
                "input_ids": encode["input_ids"],
                "attention_mask": encode["attention_mask"],
                "labels": label_ids
            }

        elif model_type == "chatGLM2":
            prompt_ids = tokenizer.encode(examples["final_prompt"])
            encode = tokenizer(examples["final_prompt"], examples[prompt_fields[-1]])

            label_ids = copy.deepcopy(encode["input_ids"])
            label_ids = [-100] * (len(prompt_ids) - 1) + label_ids[len(prompt_ids) - 1:]

            feature = {
                "input_ids": encode["input_ids"],
                "attention_mask": encode["attention_mask"],
                "labels": label_ids
            }

        else:
            raise ValueError("No support model type: %s" % model_type)
        return feature

    def transform_for_inference(examples):
        if model_type == "chatGLM":
            encode = tokenizer(
                examples["final_prompt"], truncation=True, max_length=max_length
            )

            feature = {
                "input_ids": encode["input_ids"],
                "position_ids": encode["position_ids"],
                "attention_mask": encode["attention_mask"]
            }

        elif model_type == "LLaMA" or model_type == "Baichuan" or model_type == "chatGLM2":
            tokenizer.add_eos_token = False
            encode = tokenizer(examples["final_prompt"])

            feature = {
                "input_ids": encode["input_ids"],
                "attention_mask": encode["attention_mask"]
            }

        else:
            raise ValueError("No support model type: %s" % model_type)
        return feature

    with accelerator.main_process_first():
        data_set = data_set.map(
            build_prompt
        )
        if stream:
            logger.info("Data examples: %s" % "\n".join(
                [json.dumps(x, ensure_ascii=False, indent=4) for x in data_set.take(3)]
            ))
        else:
            logger.info("Data examples: %s" % json.dumps(data_set[:3], ensure_ascii=False, indent=4))
        if inference:
            data_set = data_set.map(
                transform_for_inference,
                batched=True,
                remove_columns=prompt_fields + ["final_prompt"]
            )
        else:
            data_set = data_set.map(
                transform,
                remove_columns=prompt_fields + ["final_prompt"]
            )

    def collate_fn(examples):
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt"
        )

    data_loader = DataLoader(
        data_set,
        collate_fn=collate_fn,
        batch_size=batch_size
    )

    return data_loader
