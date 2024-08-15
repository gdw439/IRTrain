from index import ModelEmbed
import jsonlines
import hashlib
from tqdm import tqdm

text_tab = {}
with jsonlines.open("/data/public/data_20240710/dataset/medi2-data-jsonl/cool_0621/train.jsonl", 'r') as f:
    cnt = 0
    for d in f:
        q, pd, nds = d['query'][-1], d['pos'][-1], d['neg'][1:]
        for item in [q, pd] + nds:
            md5 = hashlib.md5(item.encode()).hexdigest()
            text_tab[md5] = item
        cnt += 1
        if cnt == 8192:
            break

md5_list, txt_list, emb_list = [], [], []
for key, val in text_tab.items():
    md5_list.append(key)
    txt_list.append(val)

batch = 1024

print(md5_list[:8])
print(txt_list[:8])

encoder = ModelEmbed("/home/guodewen/research/IRTrain/models/stella_v2_large", batch_size=1024)
for pos in tqdm(range(0, len(text_tab), batch), desc="embedding process"):
    batch_data = txt_list[pos: pos + batch]
    emb_list.append(encoder.emdbed(batch_data).to('cpu'))

import numpy as np
np.save('md5.npy', md5_list)

import torch
emb_torch = torch.vstack(emb_list)
torch.save(emb_torch, 'emb.pt')
