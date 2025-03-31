import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import load_dataset
from utils.dp_optimizer import DPAdam_Optimizer
from utils.sampling import get_data_loaders_possion

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default="./model")

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--iter', type=int, default=500)

parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--C', type=float, default=0.1)
parser.add_argument('--epsilon', type=float, default=3.0)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda',choices=['cpu', 'cuda'])

args = parser.parse_args()
###
dataset = load_dataset("text", data_files="processed_data.txt")
train_dataset = dataset["train"]
###

### Load Model and Tokenizer
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
###


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
    params=model.parameters(),
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

for i in range(args.iter):
    train_dl = minibatch_loader(train_dataset)
    model.train()
    loss = 0.0
    for id, batch in enumerate(train_dl):
        batch_len = len(batch['input_ids'])
        optimizer.minibatch_size = batch_len
        optimizer.zero_accum_grad()
        for iid in range(batch_len):
            optimizer.zero_microbatch_grad()

            input_ids_sample = batch["input_ids"][0]
            attention_mask_sample = batch["attention_mask"][0]
            labels_sample = batch["labels"][0]
            sample= {
                "input_ids": input_ids_sample.unsqueeze(0),
                "attention_mask": attention_mask_sample.unsqueeze(0),
                "labels": labels_sample.unsqueeze(0),
            }

            output = model(**sample)
            labels = sample['input_ids'][:, 1:, ]
            logits = output.logits[:, :-1, :].permute(0, 2, 1)
            sample_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(dim=1)
            sample_loss.backward()
            loss += sample_loss.item()
            optimizer.microbatch_step()
        optimizer.step_dp()
    loss /=batch_len

    if loss < least_loss:
        least_loss = loss
        model.save_pretrained(least_loss_dir)
        tokenizer.save_pretrained(least_loss_dir)

    print(f'iters:{i}, |'f' Average loss: {loss:.4f}')


model.save_pretrained("fine_tuned_gpt2_dp_2")
tokenizer.save_pretrained("fine_tuned_gpt2_dp_2")

print("Train Completed, Fine Tuned Model parameters have been stored in fine_tuned_gpt2_dp_2")
