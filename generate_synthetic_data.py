import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2_dp")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_gpt2_dp")

done_counter = 0

def generate_text(prompt, max_length = 128):
    global done_counter
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    model.eval()
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask = attention_mask, max_length = max_length, pad_token_id = tokenizer.eos_token_id,  do_sample=True, top_k=50, top_p=0.95)
    print(f"done{done_counter}")
    done_counter += 1
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    cutoff_index = generated_text.find("50K")
    if cutoff_index != -1:
        generated_text = generated_text[:cutoff_index + 3]

    return generated_text

ids = [random.randint(1000, 9999) for _ in range(300)]
generate_texts = [generate_text(f"ID: {ids[i]},") for i in range(0, 300)]

output_file = "synthetic_data_dp.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for text in generate_texts:
        f.write(text + "\n")

print(f"Synthetic Data has been stored in {output_file}")