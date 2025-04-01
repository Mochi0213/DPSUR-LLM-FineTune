import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import load_dataset
from utils.dp_optimizer import DPAdam_Optimizer
from utils.sampling import get_data_loaders_possion
from peft import get_peft_model, LoraConfig, TaskType
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default="./model")

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--iter', type=int, default=500)

parser.add_argument('--sigma', type=float, default=1.23)
parser.add_argument('--C', type=float, default=0.1)
parser.add_argument('--epsilon', type=float, default=3.0)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda',choices=['cpu', 'cuda'])
parser.add_argument('--model', type=str, default='distilgpt2', choices = ['distilgpt2', 'gpt2-large'])
parser.add_argument('--algorithm', type=str, default='DPSGD', choices=['DPSGD', 'DPSUR'])
args = parser.parse_args()
###
dataset = load_dataset("text", data_files="processed_data.txt")
train_dataset = dataset["train"]
###

### Load Model and Tokenizer
model = GPT2LMHeadModel.from_pretrained(args.model)
tokenizer = GPT2Tokenizer.from_pretrained(args.model)
###

lora_config = LoraConfig(
    r=8,                             # 低秩矩阵的秩
    lora_alpha=16,                   # 缩放因子
    target_modules=["c_attn", "c_proj"],  # GPT2 中的关键全连接层
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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
    l2_norm_clip=args.C,
    noise_multiplier=args.sigma,
    minibatch_size=args.batch_size,
    microbatch_size=1,
    params = filter(lambda p: p.requires_grad, model.parameters()),
    lr = args.lr
    )

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

minibatch_loader, microbatch_loader = get_data_loaders_possion(
    minibatch_size=args.batch_size, 
    microbatch_size=1,
    iterations=1,
    collate_fn=data_collator
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
least_loss_dir = './least_loss_model'
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
iter = 1
epsilon = 0.0

while epsilon < args.epsilon:
    
    epsilon, best_alpha = apply_dp_sgd_analysis(args.batch_size / len(train_dataset), args.sigma, iter, orders, args.delta) #comupte privacy cost
    train_dl = minibatch_loader(train_dataset)
    model.train()
    loss = 0.0
    for id, batch in enumerate(train_dl):
        batch_len = len(batch['input_ids'])
        optimizer.minibatch_size = batch_len
        optimizer.zero_accum_grad()
        for iid in range(batch_len):
            optimizer.zero_microbatch_grad()

            input_ids_sample = batch["input_ids"][iid]
            attention_mask_sample = batch["attention_mask"][iid]
            labels_sample = batch["labels"][iid]
            sample= {
                "input_ids": input_ids_sample.unsqueeze(0),
                "attention_mask": attention_mask_sample.unsqueeze(0),
                "labels": labels_sample.unsqueeze(0),
            }

            output = model(**sample)
            labels = sample['input_ids'][:, 1:, ]
            logits = output.logits[:, :-1, :].permute(0, 2, 1)
            # sample_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(dim=1)
            
            ### update sample loss caculation
            attention_mask = sample["attention_mask"][:, 1:]
            token_losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            masked_losses = token_losses * attention_mask
            sample_loss = masked_losses.sum(dim=1) / attention_mask.sum(dim=1)
            ###            
            
            sample_loss.backward()
            loss += sample_loss.item()
            optimizer.microbatch_step()
        optimizer.step_dp()
    loss /=batch_len

    if loss < least_loss:
        least_loss = loss
        # model.save_pretrained(least_loss_dir)
        # tokenizer.save_pretrained(least_loss_dir)

    print(f'iters:{iter}, 'f'epsilon:{epsilon:.4f} |'f' Average loss: {loss:.4f}')
    iter+=1


model.save_pretrained("fine_tuned_gpt2_dp_lora")
tokenizer.save_pretrained("fine_tuned_gpt2_dp_lora")

print("Train Completed, Fine Tuned Model parameters have been stored in fine_tuned_gpt2_dp_2")
