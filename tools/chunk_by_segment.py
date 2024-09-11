import jsonlines
import re
from tqdm import tqdm

with jsonlines.open("/home/guodewen/research/IRTrain/dataset/IR-TEST/govern/govern-content-uniq.jsonl") as fr:
    lines = list(fr)

with jsonlines.open("/home/guodewen/research/IRTrain/dataset/IR-TEST/govern/govern-content-uniq-seg.jsonl", "w", flush=True) as fw:
    for line in tqdm(lines):
        cache, slices, content = "", [], line['content']
        # sentences = re.findall(r'.+?。', content)
        sentences = re.split(r'(。)', content)
        for sent in sentences:
            if len(cache) <= 512:
                cache += sent
            else:
                slices.append(cache.lstrip("。"))
                cache = sent
        if cache != "":
            slices.append(cache.lstrip("。"))
        line['slices'] = slices
        fw.write(line)