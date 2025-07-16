from transformers import AutoModelForCausalLM,AutoTokenizer
from  datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from modelscope.utils.hub import snapshot_download
import torch

model_path = snapshot_download(
            repo_id='AI-ModelScope/gpt-j-6b',
            cache_dir='./gpt-j-6b'
        )

model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                weights_only=False
)

tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
from torch import nn



torch.cuda.empty_cache()
torch.cuda.synchronize()


def load_and_tokenize_dataset(cache_dir: str, tokenizer, batch_size: int = 1):

    ds = load_dataset("JeanKaddour/minipile", split="validation", cache_dir=cache_dir)

    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Group the dataset into blocks of model_max_length
    block_size = tokenizer.model_max_length
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


dataloader=load_and_tokenize_dataset("./minipile_cache",tokenizer,1)


# model.to('cpu')

prompt='China is a'
inputs = tokenizer(prompt, return_tensors='pt')
model=model.to('cuda:0')
output=model(input_ids=inputs['input_ids'].to('cuda:0'))
logits=output.logits
predicted_ids = torch.argmax(logits, dim=-1)
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0])  # 获取第一批次的词索引并转换为词

# 3. 输出生成的文本
generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print(generated_text)
torch.cuda.empty_cache()
torch.cuda.synchronize()

print(tokenizer.pad_token_id)
criterion = nn.CrossEntropyLoss(ignore_index=-100,reduction='mean')


from tqdm import tqdm
device='cuda:0'    # Evaluation loop
total_loss = 0.0
total_batches = 0
# torch.cuda.empty_cache()
# torch.cuda.synchronize()
model.eval()
# model=model.to(device)
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating"):
        # 拿到完整的 input_ids, attention_mask, 和已经被 collator 设好 -100 的 labels
        input_ids    = batch['input_ids'].to(device)       # [B, T]
        attention_mask = batch['attention_mask'].to(device)# [B, T]
        labels       = batch['labels'].to(device)          # [B, T], pad 已经是 -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids,)
                            # attention_mask=attention_mask)
            logits  = outputs.logits                     # [B, T, V]

        # 手动 shift：logits 丢掉最后一位，labels 丢掉第一位
        shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

        # 计算交叉熵 loss，ignore_index=-100 会跳过所有 pad 位置
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),  # [(B*(T-1)), V]
            shift_labels.view(-1)                          # [(B*(T-1))]
        )
        print(loss.item)
        total_loss   += loss.item()
        total_batches+= 1

avg_loss = total_loss / total_batches
# perplexity = math.exp(avg_loss)