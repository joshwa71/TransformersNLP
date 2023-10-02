from modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import numpy as np
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from datasets import load_dataset, load_metric
from datetime import datetime

checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_datasets = load_dataset("glue", "sst2")

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx", "label"])
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=32, collate_fn=data_collator)
model = Decoder(vocab_size=tokenizer.vocab_size, max_len=tokenizer.max_model_input_sizes[checkpoint], d_k=64, d_model=256, n_heads=4, n_layers=4, dropout_prob=0.1)                                 
model.to('cuda')
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, train_dataloader, epochs):
    for epoch in range(epochs):
        model.train()
        t0 = datetime.now()
        train_losses = []
        for batch in train_dataloader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            optimizer.zero_grad()

            targets = batch["input_ids"].clone().detach()
            targets = torch.roll(targets, shifts=-1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id
            #targets = targets.to('cuda')

            outputs = model(batch["input_ids"], batch['attention_mask'])
            loss = criterion(outputs.transpose(2, 1), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        train_losses[epoch] = train_loss
        print(f"Epoch {epoch+1}/{epochs} train loss: {train_loss} time: {datetime.now() - t0}")
    
    return train_losses

train(model, criterion, optimizer, train_dataloader, 5)




        
