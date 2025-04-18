import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import load_dataset
from utils.dp_optimizer import DPAdam_Optimizer
from utils.sampling import get_data_loaders_possion
from peft import get_peft_model, LoraConfig, TaskType
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.RDP.compute_rdp import compute_rdp
import copy
import numpy as np

import json
import os
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default="./model")

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--model', type=str, default='gpt2-large', choices=['distilgpt2', 'gpt2-large', 'gpt2-xl'])
parser.add_argument('--algorithm', type=str, default='DPSGD', choices=['DPSGD', 'DPSUR'])

### DP Parameters
parser.add_argument('--epsilon', type=float, default=3.0)

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--bs_valid', type=int, default=10)

parser.add_argument('--sigma_t', type=float, default=1.25)
parser.add_argument('--sigma_v', type=float, default=1.5)

parser.add_argument('--C_t', type=float, default=0.1)
parser.add_argument('--C_v', type=float, default=0.001)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--beta', type=float, default=-1.0)
###

args = parser.parse_args()
###
dataset = load_dataset("text", data_files="processed_data_2000.txt")
train_dataset = dataset["train"]
###

### Load Model and Tokenizer
model = GPT2LMHeadModel.from_pretrained(args.model)
tokenizer = GPT2Tokenizer.from_pretrained(args.model)

# model = GPT2LMHeadModel.from_pretrained('gpt2-large')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
###

lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=16,  # 缩放因子
    target_modules=["c_attn", "c_proj"],  # GPT2 中的关键全连接层
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(args.device)

### Tokenizing train_dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


tokenizer.pad_token = tokenizer.eos_token

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
###
print("train_dataset包含总样本数:", len(train_dataset))
print(train_dataset[0])
###

### DP optimizer
optimizer = DPAdam_Optimizer(
    l2_norm_clip=args.C_t,
    noise_multiplier=args.sigma_t,
    minibatch_size=args.batch_size,
    microbatch_size=1,
    params=filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)



class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[int(idx)]

    def __len__(self):
        return len(self.dataset)

train_dataset = WrappedDataset(train_dataset)

least_loss = 99999.0
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
iter = 1
epsilon = 0.0

epsilon_list = []
loss_list = []

if args.algorithm == 'DPSGD':
    minibatch_loader, microbatch_loader = get_data_loaders_possion(
        minibatch_size=args.batch_size,
        microbatch_size=1,
        iterations=1,
        collate_fn=data_collator
    )
    minibatch_loader_for_test, microbatch_loader_for_test = get_data_loaders_possion(
        minibatch_size=40,
        microbatch_size=1,
        iterations=1,
        collate_fn=data_collator
    )
    while epsilon < args.epsilon:

        epsilon, best_alpha = apply_dp_sgd_analysis(args.batch_size / len(train_dataset), args.sigma_t, iter, orders,
                                                    args.delta)  # comupte privacy cost
        train_dl = minibatch_loader(train_dataset)
        model.train()
        loss = 0.0
        batch_len = 0.0
        for id, batch in enumerate(train_dl):
            batch_len = len(batch['input_ids'])
            optimizer.minibatch_size = batch_len
            optimizer.zero_accum_grad()
            for iid in range(batch_len):
                optimizer.zero_microbatch_grad()

                input_ids_sample = batch["input_ids"][iid].to(args.device)#fatal error fixed
                attention_mask_sample = batch["attention_mask"][iid].to(args.device)
                labels_sample = batch["labels"][iid].to(args.device)
                sample = {
                    "input_ids": input_ids_sample.unsqueeze(0),
                    "attention_mask": attention_mask_sample.unsqueeze(0),
                    "labels": labels_sample.unsqueeze(0),
                }
                # print(sample)
                output = model(**sample)
                labels = sample['input_ids'][:, 1:, ]
                logits = output.logits[:, :-1, :].permute(0, 2, 1)

                sample_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(dim=1)
                
                ### update sample loss caculation
                # attention_mask = sample["attention_mask"][:, 1:]
                # token_losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                # masked_losses = token_losses * attention_mask
                # sample_loss = masked_losses.sum(dim=1) / attention_mask.sum(dim=1)
                ###
                
                sample_loss.backward()
                # print('the sample loss is:', sample_loss.item())
                loss += sample_loss.item()
                optimizer.microbatch_step()
            optimizer.step_dp()

        model.eval()
        test_dl = minibatch_loader_for_test(train_dataset)
        test_loss = 0

        with torch.no_grad():
            for id, batch in enumerate(test_dl):
                batch.to(args.device)
                outputs = model(**batch)
                labels = batch['input_ids'][:, 1:, ].to(args.device)
                logits= outputs.logits[:, :-1, :].permute(0, 2, 1).to(args.device)
                test_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none') \
                    .mean(dim=1) \
                    .mean()
        print(f'iters:{iter}, 'f'epsilon:{epsilon:.4f} |'f' Average loss: {test_loss:.4f}')
        epsilon_list.append(epsilon)
        loss_list.append(test_loss)
        iter += 1

elif args.algorithm == 'DPSUR':
    t = 1
    last_valid_loss = 99999.0

    last_model = model
    minibatch_loader_for_train, microbatch_loader_for_train = get_data_loaders_possion(
        minibatch_size=args.batch_size,
        microbatch_size=1,
        iterations=1,
        collate_fn=data_collator
    )
    minibatch_loader_for_valid, microbatch_loader_for_vaild = get_data_loaders_possion(
        minibatch_size=args.bs_valid,
        microbatch_size=1,
        iterations=1,
        collate_fn=data_collator
    )
    minibatch_loader_for_test, microbatch_loader_for_test = get_data_loaders_possion(
        minibatch_size=40,
        microbatch_size=1,
        iterations=1,
        collate_fn=data_collator
    )
    while epsilon < args.epsilon:
        rdp_train = compute_rdp(args.batch_size / len(train_dataset), args.sigma_t, t, orders)
        rdp_valid = compute_rdp(args.bs_valid / len(train_dataset), args.sigma_v, t, orders)
        epsilon, best_alpha = compute_eps(orders, rdp_train + rdp_valid, args.delta)

        train_dl = minibatch_loader_for_train(train_dataset)
        valid_dl = minibatch_loader_for_valid(train_dataset)
        ###train
        model.train()
        train_loss = 0.0
        batch_len = 0.0
        for id, batch in enumerate(train_dl):
            batch_len = len(batch['input_ids'])
            optimizer.minibatch_size = batch_len
            optimizer.zero_accum_grad()
            for iid in range(batch_len):
                optimizer.zero_microbatch_grad()

                input_ids_sample = batch["input_ids"][iid].to(args.device)
                attention_mask_sample = batch["attention_mask"][iid].to(args.device)
                labels_sample = batch["labels"][iid].to(args.device)
                sample = {
                    "input_ids": input_ids_sample.unsqueeze(0),
                    "attention_mask": attention_mask_sample.unsqueeze(0),
                    "labels": labels_sample.unsqueeze(0),
                }
                # print(sample)
                output = model(**sample)
                labels = sample['input_ids'][:, 1:, ]
                logits = output.logits[:, :-1, :].permute(0, 2, 1)
                sample_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(dim=1)
                
                ### update sample loss caculation
                # attention_mask = sample["attention_mask"][:, 1:]
                # token_losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                # masked_losses = token_losses * attention_mask
                # sample_loss = masked_losses.sum(dim=1) / attention_mask.sum(dim=1)
                ###
                
                sample_loss.backward()
                train_loss += sample_loss.item()
                optimizer.microbatch_step()
            optimizer.step_dp()
        ###train
        ###test
        test_loss = 0
        test_dl = minibatch_loader_for_test(train_dataset)
        with torch.no_grad():
            for id, batch in enumerate(test_dl):
                batch = batch.to(args.device)
                outputs = model(**batch)
                labels = batch['input_ids'][:, 1:, ]
                logits= outputs.logits[:, :-1, :].permute(0, 2, 1)
                test_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none') \
                    .mean(dim=1) \
                    .mean()
        print(f'iters:{iter}, 'f'epsilon:{epsilon:.4f} |'f' Average loss: {test_loss:.4f}')
        ###test
        ###vailid
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for id, batch in enumerate(valid_dl):
                batch = batch.to(args.device)
                outputs_new = model(**batch)
                outputs_last = last_model(**batch)
                labels = batch['input_ids'][:, 1:, ].to(args.device)
                logits_new = outputs_new.logits[:, :-1, :].permute(0, 2, 1).to(args.device)
                logits_last = outputs_last.logits[:, :-1, :].permute(0, 2, 1).to(args.device)
                valid_loss_new = torch.nn.functional.cross_entropy(logits_new, labels, reduction='none')\
                    .mean(dim=1)\
                    .mean()
                valid_loss_last = torch.nn.functional.cross_entropy(logits_last, labels, reduction='none') \
                    .mean(dim=1) \
                    .mean()
        ###valid
        deltaE = valid_loss_new - valid_loss_last
        deltaE = torch.tensor(deltaE).cpu()
        print("Delta E:", deltaE)
        deltaE = np.clip(deltaE, -args.C_v, args.C_v)
        deltaE_after_dp = 2*args.C_v*args.sigma_v*np.random.normal(0,1)+deltaE
        print("Delta E after dp:",deltaE_after_dp)
        if deltaE_after_dp < args.beta*args.C_v:
            last_valid_loss = valid_loss_new
            last_model = copy.deepcopy(model)
            epsilon_list.append(epsilon)
            loss_list.append(test_loss)
            print("accept this round's update, the number of total accepted updates is:", format(t))
            t += 1
        else:
            print("reject this round's update")
            model.load_state_dict(last_model.state_dict(), strict=True)

        iter+=1


timestamp = datetime.now().strftime("%Y.%m%d.%H.%M")

filename = f"{args.algorithm}, eps={args.epsilon}, batch_size={args.batch_size}, bs_valid={args.bs_valid}, sigma_t={args.sigma_t}, sigma_v={args.sigma_v}, {timestamp}.json"
save_path = os.path.join("experiments", filename)

# 准备要保存的实验数据
experiment_data = {
    "args": vars(args),  # 所有超参数
    "epsilon_list": epsilon_list,
    "loss_list": [float(loss) for loss in loss_list]
}

# 写入 JSON 文件
with open(save_path, "w") as f:
    json.dump(experiment_data, f, indent=4)

print(f"Experiment results saved to {save_path}")
