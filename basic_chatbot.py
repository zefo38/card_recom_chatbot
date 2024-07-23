import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from Dataset import ChatbotDataset
import re
import argparse

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default = 3e-5, type = float)
parser.add_argument("--epoch", default = 10, type = int)
parser.add_argument("--Sneg", default = -1e18, type = float)
args = parser.parse_args('')

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token = BOS, eos_token = EOS, unk_token = '<unk>',
                                                    pad_token = PAD, mask_token = MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

data = pd.read_csv('./ChatBotData.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = ChatbotDataset(data, max_len = 40)

train_dataloader = DataLoader(train_set, batch_size = 32, num_workers = 0, shuffle = True, collate_fn = ChatbotDataset.collate_batch,)

model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction = "none")
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


print("start")
for epoch in range(args.epoch):
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        token_ids, mask, label = token_ids.to(device), mask.to(device), label.to(device)
        out = model(token_ids)
        out = out.logits
        mask_3d = mask.unsqueeze(dim = 2).repeat_interleave(repeats = out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, args.Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        optimizer.step()
    print(f"epoch {epoch} loss {avg_loss}")
    model.save_pretrained("./basic_model")
print("end")
