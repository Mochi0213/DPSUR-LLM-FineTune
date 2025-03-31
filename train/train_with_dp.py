import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F

import numpy as np
from transformers import GPT2LMHeadModel


def train_with_dp(model, train_loader, optimizer,device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp()


    return train_loss, train_acc





def train_with_dp_gpt2(model, train_loader, optimizer, tokenizer, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        optimizer.zero_accum_grad()

        for i in range(len(input_ids)):  # microbatch
            optimizer.zero_microbatch_grad()

            output = model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0),
                           labels=input_ids[i].unsqueeze(0))
            loss = output.loss  # GPT-2 自带 loss 计算

            loss.backward()
            optimizer.microbatch_step()

        optimizer.step_dp()
        total_loss += loss.item()

    return total_loss / len(train_loader)  # 返回平均 loss