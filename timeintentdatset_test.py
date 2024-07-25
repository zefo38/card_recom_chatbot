import pandas as pd
import numpy as np
from TimeIntentDataset import timeintentdataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
import os
import json


path = "./Training/labeled"
files = os.listdir(path)
files
js_list = []

for file in files:
    with open(os.path.join(path, file), encoding = 'utf-8') as f:
        js = json.load(f)
    js_list.append(js)
    
print(js_list)

BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token = BOS, eos_token = EOS, unk_token = '<unk>',
                                                    pad_token = PAD, mask_token = MASK)

data = js_list
train_set = timeintentdataset(data, tokenizer, max_length = 128)
train_loader = DataLoader(train_set, batch_size = 32, num_workers = 0, shuffle = True)
print(train_loader)