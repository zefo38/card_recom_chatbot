import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import json

BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
SENT = '<unused1>'
MASK = "<unused0>"
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK,)

class timeintentdataset(Dataset):
    def __init__(self, file, tokenizer, max_length = 128):
        self._data = file
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        utterance = self._data[idx]
        text = utterance['text']
        intent = utterance['timex3']["temp_s_id"].sum()
        if intent >= 0:
            intent = 0
        inputs = self.tokenizer(text, truncation = True, padding = self.max_length, return_tensors = 'pt')
        item = {key:val.squeeze() for key, val in inputs.items()}
        item['labels'] = torch.tensor(int(intent), dtype = torch.long)
        return item