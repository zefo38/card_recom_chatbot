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

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

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

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction = "none")
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

epoch = 10
Sneg = -1e18

print("start")
for epoch in range(epoch):
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        samples = samples.to(device)
        out = model(token_ids)
        out = out.logits
        mask_3d = mask.unsqueeze(dim = 2).repeat_interleave(repeats = out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        optimizer.step()
print("end")