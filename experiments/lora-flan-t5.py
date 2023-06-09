import math
import json
import random
from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup


import wandb


class LoRALayer(nn.Module):
    
    def __init__(self, layer: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        
        self.layer = layer
        
        self.rank = rank
        self.alpha = alpha
        
        self.lora_a = nn.Parameter(self.layer.weight.new_zeros((self.rank, self.layer.in_features)))
        self.lora_b = nn.Parameter(self.layer.weight.new_zeros((self.layer.out_features, self.rank)))

        self.scaling = self.alpha / self.rank
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.post_init()
        
    def post_init(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        
    def forward(self, x: torch.Tensor):
        x_lora = self.layer(x) + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T * self.scaling)
        return x_lora


# model_name = "google/t5-xl-lm-adapt"
model_name = "google/flan-t5-xl"

num_epochs = 8
batch_size = 12
grad_acum_steps = 10

save_each_steps = 500

learning_rate = 0.0002

num_warmup_steps = 500

lora_dropout = 0.1
alpha = 32
rank = 8

lora_lm_head = False

use_scheduler = True


def prompting(dialog):

    prompted_dialog = list()

    for n, phrase in enumerate(dialog[::-1]):
        person_prompt = "Person: " if n % 2 == 0 else "Bot: "
        prompted_phrase = person_prompt + phrase
        prompted_dialog.insert(0, prompted_phrase)

    prompted_dialog.append("Bot:")

    return prompted_dialog


class SequenceDataset(Dataset):

    def __init__(self, data):
        super().__init__()

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample = self.data[index]

        context = "\n".join(prompting(sample["dialog"]))
        
        response = "<pad>" + sample["response"]

        return context, response


class Collator:

    def __init__(self, tokenizer, encoder_max_length=1024, decoder_max_length=128):

        self.tokenizer = tokenizer

        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

    def __call__(self, batch):

        contexts = list()
        responses = list()

        for context, response in batch:
            contexts.append(context)
            responses.append(response)

        tokenized_contexts = self.tokenizer(
            contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.encoder_max_length
        )

        tokenized_responses = self.tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.decoder_max_length
        )

        tokenized_contexts["decoder_input_ids"] = tokenized_responses["input_ids"][:, :-1]
        tokenized_contexts["decoder_attention_mask"] = tokenized_responses["attention_mask"][:, :-1]

        targets = tokenized_responses["input_ids"][:, 1:]

        return tokenized_contexts, targets


train = list()

with open("./category_5_cleaned.jsonl") as file_object:
    for line in file_object:
        train.append(json.loads(line))


random.shuffle(train)


tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

for param in model.parameters():
    param.requires_grad = False
    
lora_parameters = {"alpha": alpha, "dropout": lora_dropout}
    
for encoder_layer in model.encoder.block:
    encoder_layer.layer[0].SelfAttention.q = LoRALayer(encoder_layer.layer[0].SelfAttention.q, rank=rank, **lora_parameters)
    encoder_layer.layer[0].SelfAttention.k = LoRALayer(encoder_layer.layer[0].SelfAttention.k, rank=rank, **lora_parameters)
    encoder_layer.layer[0].SelfAttention.v = LoRALayer(encoder_layer.layer[0].SelfAttention.v, rank=rank, **lora_parameters)
    encoder_layer.layer[0].SelfAttention.o = LoRALayer(encoder_layer.layer[0].SelfAttention.o, rank=rank, **lora_parameters)
    
for decoder_layer in model.decoder.block:
    decoder_layer.layer[0].SelfAttention.q = LoRALayer(decoder_layer.layer[0].SelfAttention.q, rank=rank, **lora_parameters)
    decoder_layer.layer[0].SelfAttention.k = LoRALayer(decoder_layer.layer[0].SelfAttention.k, rank=rank, **lora_parameters)
    decoder_layer.layer[0].SelfAttention.v = LoRALayer(decoder_layer.layer[0].SelfAttention.v, rank=rank, **lora_parameters)
    decoder_layer.layer[0].SelfAttention.o = LoRALayer(decoder_layer.layer[0].SelfAttention.o, rank=rank, **lora_parameters)
    
#     decoder_layer.layer[1].EncDecAttention.q = LoRALayer(decoder_layer.layer[1].EncDecAttention.q, rank=rank, **lora_parameters)
#     decoder_layer.layer[1].EncDecAttention.k = LoRALayer(decoder_layer.layer[1].EncDecAttention.k, rank=rank, **lora_parameters)
#     decoder_layer.layer[1].EncDecAttention.v = LoRALayer(decoder_layer.layer[1].EncDecAttention.v, rank=rank, **lora_parameters)
#     decoder_layer.layer[1].EncDecAttention.o = LoRALayer(decoder_layer.layer[1].EncDecAttention.o, rank=rank, **lora_parameters)
    
if lora_lm_head:
    model.lm_head = LoRALayer(model.lm_head, rank=rank)

    
config = {
    "model_name": model_name,
    "num_warmup_steps": num_warmup_steps,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "grad_acum_steps": grad_acum_steps,
    "rank": rank,
    "alpha": alpha,
    "lora_dropout": lora_dropout,
    "lora_lm_head": lora_lm_head,
    "use_scheduler": use_scheduler,
    "num_parameters": sum([p.numel() for p in model.parameters()]),
    "num_trainable_parameters": sum([p.numel() for p in model.parameters() if p.requires_grad]),
}

wandb.init(config=config, project="t5-lora")

model.cuda()

# model.load_state_dict(torch.load("./t5-xl-lm-adapt_0_3_classes_lora.pt"))

train_dataset = SequenceDataset(data=train)

collator = Collator(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)


optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)


if use_scheduler:
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_epochs * len(train_loader)
    )
else:
    scheduler = None


def loop(n_epoch, is_train, loader, grad_acum_steps=1):
    
    global_step = len(loader) // grad_acum_steps * n_epoch

    if is_train:
        model.train()
    else:
        model.eval()

    all_predictions = list()
    all_targets = list()

    losses = list()

    progress_bar = tqdm(total=len(loader) // grad_acum_steps, desc="Train" if is_train else "Valid")

    if is_train:
        model.train()
    else:
        model.eval()

    losses = list()

    for n_step, (batch, targets) in enumerate(loader):

        batch = batch.to(model.device)
        targets = targets.to(model.device)

        if is_train:
            logits = model(**batch).logits
        else:
            with torch.inference_mode():
                logits = model(**batch).logits

        loss = criterion(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))

        losses.append(loss.item())

        if is_train:
            loss.backward()
            if n_step > 0 and n_step % grad_acum_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.5)
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
                progress_bar.update()
                progress_bar.set_postfix(loss=np.mean(losses[-100:]))
                wandb.log({"train_loss": loss.item()})
                
#                 if global_step > 0 and global_step % save_each_steps == 0:
#                     torch.save(model.state_dict(), f"./{model_name.split('/')[-1]}_{global_step}_steps_5_cat_lora.pt")

                global_step += 1
        else:
            progress_bar.update()
            progress_bar.set_postfix(loss=np.mean(losses[-100:]))

    progress_bar.close()

    return losses


wandb.watch(model, log_freq=100)

for n_epoch in range(num_epochs):

    train_losses = loop(n_epoch, is_train=True, loader=train_loader, grad_acum_steps=grad_acum_steps)

    train_mean_loss = np.mean(train_losses)
 
    wandb.log({"epoch_train_loss": train_mean_loss})

    epoch_message = [
        f"Epoch {n_epoch} done",
        "",
        "Train",
        f"\tLoss: {train_mean_loss:.3f}",
    ]

    print("\n".join(epoch_message))

    torch.save(model.state_dict(), f"./{model_name.split('/')[-1]}_{n_epoch}_epoch_5_category_lora.pt")
