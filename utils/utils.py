from transformers import AutoModelForCausalLM, AutoTokenizer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random
import string

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
INSTRUCT = "Answer the question concisely. Q: {} A:"


MODEL_NAME = {
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-v0.2": "mistralai/Mistral-7B-Instruct-v0.2", 
    "mistral-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2": "google/gemma-2-9b-it",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5_32b": "Qwen/Qwen2.5-32B-Instruct",
    "vicuna-v1.5": "lmsys/vicuna-7b-v1.5",
    "llama3.1":"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2":"meta-llama/Llama-3.2-3B-Instruct",
}


def get_model_name(model_name):
    return MODEL_NAME[model_name]

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
# def load_model_and_tokenizer(model_name, device, torch_dtype=torch.float16):
#     """prepare LLM and tokenizer"""
#     model_name = get_model_name(model_name)

#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype,).to(device)

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
    
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = model.config.eos_token_id
    
#     return model, tokenizer

# def load_model_and_tokenizer(model_name=None, device=None, model_dir=None, torch_dtype=torch.float16):
#     """prepare LLM and tokenizer"""
#     if model_name is not None:
#         model_name = get_model_name(model_name)

#         model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype,).to(device)

#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype,).to(device)

#         tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     # tokenizer.pad_token = tokenizer.eos_token
#     # model.config.pad_token_id = model.config.eos_token_id
    
#     return model, tokenizer
def _parse_gpus(gpus):
    if gpus is None:
        return None
    if isinstance(gpus, (list, tuple)):
        return [int(x) for x in gpus]
    if isinstance(gpus, str):
        tokens = gpus.replace(",", " ").split()
        if len(tokens) == 0:
            return None
        return [int(x) for x in tokens]
    return None

def load_model_and_tokenizer(model_name=None, device=None, model_dir=None, torch_dtype=torch.float16, gpus=None):
    load_path = get_model_name(model_name) if model_name is not None else model_dir
    gpus_list = _parse_gpus(gpus)
    if gpus_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpus_list)
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if gpus_list is not None and len(gpus_list) > 1:
        device_map = "auto"
        device = None
    else:
        device_map = "auto" if device is None else {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # 4. 关于 Padding 的配置 (虽然你只推理，但建议保留这部分以增强鲁棒性)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer
def load_bleurt(device):
    """BLEURT model and tokenizer"""
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
    model.eval()
    
    return model, tokenizer

def get_chat(model_name: str, question: str):
    """chat template for LLMs"""
    prompt = INSTRUCT.format(question)
    if "llama" or "qwen" in model_name:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    elif "mistral" in model_name or "gemma" in model_name:
        chat = [
            {"role": "user", "content": prompt},
        ]
        
    return chat
    
    
def write_to_csv(generated_sentence, label, file_path):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([generated_sentence, label]) 
        
        
        
def preprocess_tqa(ds):
    """remove the null string in 'correct_answers' and 'incorrect_answers' """
    def remove_empty_answers(example):
        example["correct_answers"] = [answer for answer in example["correct_answers"] if answer.strip()]
        example["incorrect_answers"] = [answer for answer in example["incorrect_answers"] if answer.strip()]
        return example
    
    filtered_ds = ds.map(remove_empty_answers)
    
    return filtered_ds
