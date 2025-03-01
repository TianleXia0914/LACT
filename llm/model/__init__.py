#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import os
import json

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from .baichuan.modeling_baichuan import BaiChuanForCausalLM
from llm.utils.tokenizer.tokenization_baichuan import BaiChuanTokenizer

from .chatGLM2.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLM2ForConditionalGeneration
from llm.utils.tokenizer.tokenization_chatglm2 import ChatGLMTokenizer as ChatGLM2Tokenizer

from .qwen.modeling_qwen import QWenLMHeadModel
from llm.utils.tokenizer.tokenization_qwen import QWenTokenizer

from .qwen2.modeling_qwen import QWenLMHeadModel as QWen2LMHeadModel
from .qwen2.tokenization_qwen import QWenTokenizer as QWen2Tokenizer

from .MiLM.mimodel import MiModel
from .MiLM.mitokenizer import MiTokenizer

from .chatGLM3.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLM3ForConditionalGeneration
from .chatGLM3.tokenization_chatglm import ChatGLMTokenizer as ChatGLM3Tokenizer


def get_model_and_tokenizer(model_dir, model_type=None, cache_dir=None,
                            device_map=None, load_in_8bit=False, quantization_config=None,
                            do_not_load_model=False, use_auto_class=False, torch_dtype=None):
    if model_type is None and os.path.exists(os.path.join(model_dir, "config.json")):
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                config_json = json.load(f)
            model_type = config_json["architectures"][0].lower()
        except Exception as e:
            pass


    kwargs = {}
    if use_auto_class:
        model_cls = AutoModelForCausalLM
        tokenizer_cls = AutoTokenizer
    elif "llama" in model_type:
        model_cls = LlamaForCausalLM
        tokenizer_cls = LlamaTokenizer
    elif "baichuan" in model_type:
        model_cls = BaiChuanForCausalLM
        tokenizer_cls = BaiChuanTokenizer
    elif "chatglm2" in model_type:
        model_cls = ChatGLM2ForConditionalGeneration
        tokenizer_cls = ChatGLM2Tokenizer
        kwargs["empty_init"] = False
    elif "chatglm3" in model_type:
        model_cls = ChatGLM3ForConditionalGeneration
        tokenizer_cls = ChatGLM3Tokenizer
        kwargs["empty_init"] = False
    elif "qwen2" in model_type:
        model_cls = QWen2LMHeadModel
        tokenizer_cls = QWen2Tokenizer
    elif "qwen" in model_type:
        model_cls = QWenLMHeadModel
        tokenizer_cls = QWenTokenizer
    else:
        model_cls = AutoModelForCausalLM
        tokenizer_cls = AutoTokenizer

    model = None
    if not do_not_load_model:
        model = model_cls.from_pretrained(
            model_dir,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            trust_remote_code=True,
            **kwargs
        )
    tokenizer = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if "qwen" in model_type:
            tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if model and len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
