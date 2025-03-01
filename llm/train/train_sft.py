#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm.model import get_model_and_tokenizer
from llm.data.sft_data import SFTDataSet

os.environ["WANDB_DISABLED"] = "true"

transformers.logging.set_verbosity_info()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(name=__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    model_type: str = field(default=None)


@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"nargs": "+"})
    valid_file: str = field(default=None)
    test_file: str = field(default=None, metadata={"nargs": "+"})
    max_length: int = field(default=1024)
    trunc_side: str = field(default="left")
    instruction: str = field(default=None)
    use_dialog_loss: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    device_map = None
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = get_model_and_tokenizer(
        model_dir=model_args.model_name_or_path,
        model_type=model_args.model_type,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        use_auto_class=model_args.use_auto_class
    )


    if tokenizer.pad_token is None:
        if model_args.model_type == "QWen":
            tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
        elif model_args.model_type == "Baichuan":
            tokenizer.pad_token_id = 0
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    data_set = SFTDataSet(args=data_args, tokenizer=tokenizer, model_type=model_args.model_type)

    if training_args.local_rank == 0:
        logger.info(f"Load data:\n{data_set.data_set}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_set.data_set["train"],
        eval_dataset=data_set.data_set["validation"],
        tokenizer=tokenizer,
        data_collator=data_set.collate
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
