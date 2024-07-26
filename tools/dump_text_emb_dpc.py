import jsonlines
import hashlib
import torch
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

class ModelEmbed(nn.Module):
    ''' 使用模型将文本表征为向量
    '''
    def __init__(self, path) -> None:
        super(ModelEmbed, self).__init__()
        self.model = BertModel.from_pretrained(path, trust_remote_code=True)

    def forward(self, **prein):
        model_output = self.model(**prein)
        attention_mask = prein['attention_mask']
        last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        vectors = torch.nn.functional.normalize(vectors, 2.0, dim=1)
        return vectors

model = ModelEmbed('/home/guodewen/research/IRTrain/models/stella_v2_large')
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to('cuda')

text_tab = {}
with jsonlines.open("/data/public/data_20240710/dataset/medi2-data-jsonl/cool_0621/train.jsonl", 'r') as f:
    for d in f:
        q, pd, nds = d['query'][-1], d['pos'][-1], d['neg'][1:]
        for item in [q, pd] + nds:
            md5 = hashlib.md5(item.encode()).hexdigest()
            text_tab[md5] = item
        
md5_list, txt_list, emb_list = [], [], []
for key, val in text_tab.items():
    md5_list.append(key)
    txt_list.append(val)

batch = 1024 * 2 * 8

print(md5_list[:8])
print(txt_list[:8])

tokenizer = AutoTokenizer.from_pretrained('/home/guodewen/research/IRTrain/models/stella_v2_large', trust_remote_code=True)
with torch.no_grad():
    for pos in tqdm(range(0, len(text_tab), batch), desc="embedding process"):
        prein = tokenizer.batch_encode_plus(
            txt_list[pos: pos + batch], 
            padding="longest", 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to('cuda')
        output = model(**prein)
        emb_list.append(output.to('cpu'))
        torch.save(output, f"./history/{pos}.pt")

import numpy as np
np.save('md5-3.npy', md5_list)

import torch
emb_torch = torch.vstack(emb_list)
torch.save(emb_torch, 'emb-3.pt')
