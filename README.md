Sure, here's a README for your project:

---

# Llama-2-7b Fine-Tuning Using QLoRA

## Overview

This project demonstrates how to fine-tune the Llama-2-7b model on Google Colab using QLoRA for more memory-efficient training. The goal is to transform Llama-2-7b into a powerful Python code generator. We leverage the PEFT library from the Hugging Face ecosystem to achieve this.

## Table of Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [Model Configuration](#model-configuration)
- [LoRA Configuration](#lora-configuration)
- [Training](#training)
- [Model Deployment](#model-deployment)
- [Generating Python Code](#generating-python-code)
- [Acknowledgements](#acknowledgements)

## Setup

Install the required libraries:
```bash
!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes einops wandb
```

## Dataset

Load the dataset:
```python
from datasets import load_dataset

dataset = load_dataset("flytech/python-codes-25k", split="train")
```

## Model Configuration

Import libraries and load the model:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
```

Load the tokenizer:
```python
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
```

## LoRA Configuration

Set up LoRA configuration:
```python
from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 8

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## Training

Set training arguments:
```python
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=100,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)
```

Initialize the trainer:
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)
```

Pre-process the model for stable training:
```python
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
```

Train the model:
```python
trainer.train()
```

## Model Deployment

Save the model:
```python
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained("outputs")
```

Reload the saved model:
```python
lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)
```

Push the model to Hugging Face Hub:
```python
from huggingface_hub import login

login()
model.push_to_hub("aftab007/llama2-qlora-finetunined-python-codes")
```

## Generating Python Code

Generate Python code with the fine-tuned model:
```python
text = "write prime no code in python"
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

User_Prompt = """Write a Python program to implement K-Means clustering. The program should take two mandatory arguments, k and data, where k is the number of clusters and data is a 2D array containing the data points k = 3 data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]"""
inputs = tokenizer(User_Prompt, return_tensors="pt").input_ids.to('cuda')
outputs = model.generate(inputs, max_new_tokens=500, do_sample=False, num_beams=1)
python_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Output:", python_code)
```

## Acknowledgements

This project utilizes the PEFT library and SFTTrainer from the Hugging Face ecosystem. Special thanks to the developers and the community for their contributions.

