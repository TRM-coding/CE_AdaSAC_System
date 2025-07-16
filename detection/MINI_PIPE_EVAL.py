#!/usr/bin/env python3
"""
Script to download the MiniPile dataset and evaluate a GPTJ-6B model on it.

This script:
1. Downloads the MiniPile dataset (6GB subset of The Pile) to a local cache.
2. Defines a function `evaluate_minipile_gptj` that:
   - Accepts a GPTJ-6B model instance (transformers.GPTJForCausalLM).
   - Tokenizes and groups the dataset into blocks.
   - Computes average cross-entropy loss and perplexity over the dataset.
   - Returns a dict with metrics: `avg_loss` and `perplexity`.

Usage:
    from transformers import GPTJForCausalLM
    from eval_minipile_gptj import evaluate_minipile_gptj

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    metrics = evaluate_minipile_gptj(model)
    print(metrics)
"""


import os
import math
import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn



def download_minipile(cache_dir: str = "./minipile_cache"):
    """
    Downloads (or loads from cache) the MiniPile dataset.

    Args:
        cache_dir: Directory to store dataset cache.

    Returns:
        A Hugging Face Dataset object (split='train').
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(
        "JeanKaddour/minipile", split="train", cache_dir=cache_dir
    )
    return dataset


def load_and_tokenize_dataset(cache_dir: str, tokenizer, batch_size: int = 1):
    """
    Loads and tokenizes the MiniPile dataset.

    Args:
        cache_dir: Directory where MiniPile is cached/downloaded.
        tokenizer: Tokenizer for tokenizing the dataset.
        batch_size: Batch size for evaluation.

    Returns:
        A DataLoader for the tokenized dataset.
    """
    # Load dataset
    ds = load_dataset("JeanKaddour/minipile", split="validation", cache_dir=cache_dir)

    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Group the dataset into blocks of block_size (use consistent max_length)
    block_size = 512  # Use the same as tokenization max_length
    def group_texts(examples):
        all_ids = sum(examples["input_ids"], [])
        total_len = (len(all_ids) // block_size) * block_size
        blocks = [all_ids[i:i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": blocks}

    lm_dataset = tokenized.map(group_texts, batched=True, remove_columns=["attention_mask"])

    # DataLoader setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(lm_dataset, batch_size=batch_size, collate_fn=data_collator)

    return dataloader


def evaluate_minipile_gptj(model, batch_size: int = 1, cache_dir: str = "./minipile_cache", Dataloader=None) -> dict:
    """
    Evaluates a GPTJ-6B model instance on the MiniPile dataset.

    Args:
        model: A transformers.GPTJForCausalLM instance.
        batch_size: Batch size for evaluation.
        cache_dir: Directory where MiniPile is cached/downloaded.

    Returns:
        A dict with keys:
            - "avg_loss": Average cross-entropy loss.
            - "perplexity": Exponential of the average loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and tokenize dataset
    tokenizer = model.tokenizer  # already initialized in the pipeline
    dataloader = None
    if Dataloader is None:
        dataloader = load_and_tokenize_dataset(cache_dir, tokenizer, batch_size)
    else:
        dataloader = Dataloader

    # Initialize loss function with ignore_index=-100 to skip padding tokens
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    # Evaluation loop
    total_loss = 0.0
    total_batches = 0

    # model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # 拿到完整的 input_ids, attention_mask, 和已经被 collator 设好 -100 的 labels
            input_ids    = batch['input_ids'].to(device)       # [B, T]
            attention_mask = batch['attention_mask'].to(device)# [B, T]
            labels       = batch['labels'].to(device)          # [B, T], pad 已经是 -100

            # Debug: 打印前几个batch的信息
            if batch_idx < 3:
                print(f"Batch {batch_idx}: input_ids shape={input_ids.shape}, labels shape={labels.shape}")
                print(f"Sample input_ids: {input_ids[0, :10].tolist()}")
                print(f"Sample labels: {labels[0, :10].tolist()}")
                print(f"Labels min/max: {labels.min().item()}/{labels.max().item()}")

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits  = outputs                     # [B, T, V]

            # Debug: 检查logits的形状和范围
            if batch_idx < 3:
                print(f"Logits shape: {logits.shape}")
                print(f"Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")

            # 手动 shift：logits 丢掉最后一位，labels 丢掉第一位
            shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
            shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

            # 计算交叉熵 loss，ignore_index=-100 会跳过所有 pad 位置
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),  # [(B*(T-1)), V]
                shift_labels.view(-1)                          # [(B*(T-1))]
            )
            
            # Debug: 打印loss信息
            if batch_idx < 3:
                print(f"Batch {batch_idx} loss: {loss.item():.4f}")
                
            total_loss   += loss.item()
            total_batches+= 1


        # for batch in tqdm(dataloader, desc="Evaluating"):
        #     input_ids = batch['input_ids'].to(device)
        #     # target_ids = input_ids[:, 1:].contiguous()  # 去掉第一个token，生成目标序列
        #     target_ids=batch['labels'].to(device)
        #     # input_ids = input_ids[:, :-1]  # 去掉最后一个token，作为模型的输入# For causal language modeling, the target is the input itself
        #     # Forward pass
        #     with torch.no_grad():
        #         outputs = model(input_ids=input_ids)
        #         logits = outputs

        #     # Compute loss (CrossEntropyLoss expects target to be in shape [batch_size, seq_len])
        #     # We need to flatten logits and targets to compute cross-entropy loss correctly
        #     loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        #     total_loss += loss.item()
        #     total_batches += 1

        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss)

    return {"avg_loss": avg_loss, "perplexity": perplexity}





if __name__ == "__main__":
    # Load model
    print()
    # model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    # results = evaluate_minipile_gptj(model)
    # print(f"Average Loss: {results['avg_loss']:.4f}")
    # print(f"Perplexity: {results['perplexity']:.2f}")
